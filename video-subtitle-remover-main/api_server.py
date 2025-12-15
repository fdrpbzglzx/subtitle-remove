# -*- coding: utf-8 -*-
"""
FastAPI服务实现，用于视频字幕去除
"""
import os
import tempfile
import uuid
import sys
import warnings
from enum import Enum, unique
import stat
import platform
from fsplit.filesplit import Filesplit
import onnxruntime as ort
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks,Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置基础设置
warnings.filterwarnings('ignore')
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

# 项目基础配置 - 保留原代码中的模型合并功能
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

class ModelName(str, Enum):
    default = "default"
    sttn = "sttn"
    lama = "lama"
    propainter = "propainter"

# 检查并合并模型文件 - 保留原代码中的模型合并逻辑
print("正在检查并合并模型文件...")
# 合并LAMA模型
if 'big-lama.pt' not in os.listdir(LAMA_MODEL_PATH):
    print("检测到LAMA模型未合并，正在合并...")
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)
    print("LAMA模型合并完成")

# 合并OCR检测模型
if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    print("检测到OCR模型未合并，正在合并...")
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)
    print("OCR模型合并完成")

# 合并PROPAINTER模型
if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    print("检测到PROPAINTER模型未合并，正在合并...")
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)
    print("PROPAINTER模型合并完成")

# 检查并合并FFmpeg
print("正在检查FFmpeg...")
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, 'ffmpeg', ffmpeg_bin)

if sys_str == "Windows" and 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')):
    print("检测到FFmpeg未合并，正在合并...")
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, 'ffmpeg', 'win_x64'))
    print("FFmpeg合并完成")

# 设置FFmpeg可执行权限
try:
    os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
except:
    print("设置FFmpeg权限失败，但不影响使用")

# 环境变量设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设备设置
USE_DML = False
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ONNX配置
ONNX_PROVIDERS = []
available_providers = ort.get_available_providers()
for provider in available_providers:
    if provider in ["CPUExecutionProvider"]:
        continue
    if provider in [
        "DmlExecutionProvider",         # DirectML，适用于 Windows GPU
        "ROCMExecutionProvider",        # AMD ROCm
        "MIGraphXExecutionProvider",    # AMD MIGraphX
        "VitisAIExecutionProvider",     # AMD VitisAI
        "OpenVINOExecutionProvider",    # Intel GPU
        "MetalExecutionProvider",       # Apple macOS
        "CoreMLExecutionProvider",      # Apple macOS
        "CUDAExecutionProvider",        # Nvidia GPU
    ]:
        ONNX_PROVIDERS.append(provider)

# 导入项目核心功能
from backend.main import SubtitleRemover, SubtitleDetect
from backend.config import InpaintMode, MODE as DEFAULT_MODE

# 创建FastAPI应用实例
app = FastAPI(
    title="视频字幕去除API",
    description="自动检测并去除视频中的硬字幕，支持多种修复模型",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务状态存储
tasks = {}

# 临时文件目录
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/remove-subtitle", response_model=dict)
async def remove_subtitle(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="要处理的视频文件"),
    model: ModelName = Form(ModelName.default, description="选择修复模型"), 
    auto_detect: bool = True,  # 是否自动检测字幕
    sub_area: str = None  # 手动指定字幕区域，格式: "ymin,ymax,xmin,xmax"
):
    """
    上传视频并去除字幕
    - **file**: 要处理的视频文件
    - **model**: 使用的修复模型 (default, sttn, lama, propainter)
    - **auto_detect**: 是否自动检测字幕区域
    - **sub_area**: 手动指定字幕区域，格式为"ymin,ymax,xmin,xmax"，当auto_detect=False时有效
    """
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 验证文件类型
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="只支持视频文件(mp4, avi, mov, mkv)")
    
    # 保存上传的文件
    input_path = os.path.join(TEMP_DIR, f"{task_id}_input.mp4")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 准备参数
    subtitle_area = None
    if not auto_detect and sub_area:
        try:
            ymin, ymax, xmin, xmax = map(int, sub_area.split(","))
            subtitle_area = (ymin, ymax, xmin, xmax)
        except:
            raise HTTPException(status_code=400, detail="字幕区域格式错误，应为'ymin,ymax,xmin,xmax'")
    
    # 设置任务状态
    tasks[task_id] = {
        "status": "processing",
        "input_file": file.filename,
        "output_path": None,
        "progress": 0
    }
    
    # 在后台处理任务
    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        input_path=input_path,
        model=model.value,  # 取出字符串值传递给后台函数
        subtitle_area=subtitle_area if not auto_detect else None
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "任务已开始处理，请稍后查询结果"
    }

@app.get("/api/task/{task_id}", response_model=dict)
async def get_task_status(task_id: str):
    """
    查询任务处理状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "input_file": task["input_file"],
        "progress": task["progress"],
        "output_url": f"/api/download/{task_id}" if task["status"] == "completed" else None
    }

@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """
    下载处理后的视频文件
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task["status"] != "completed" or not task["output_path"]:
        raise HTTPException(status_code=400, detail="任务尚未完成或处理失败")
    
    return FileResponse(
        path=task["output_path"],
        filename=f"{task['input_file'].split('.')[0]}_no_sub.mp4",
        media_type="video/mp4"
    )

@app.get("/api/models")
async def list_models():
    """
    获取支持的模型列表
    """
    return {
        "models": [
            {"id": "default", "name": "默认模型", "description": "根据配置文件使用默认模型"},
            {"id": "sttn", "name": "STTN模型", "description": "真人视频效果较好，速度快"},
            {"id": "lama", "name": "LAMA模型", "description": "动画视频效果好，速度一般"},
            {"id": "propainter", "name": "PROPAINTER模型", "description": "运动场景效果好，需要更多显存"}
        ]
    }

@app.post("/api/detect-subtitle")
async def detect_subtitle(file: UploadFile = File(..., description="要检测的视频文件")):
    """
    仅检测字幕区域，不进行去除处理
    """
    # 生成临时ID
    temp_id = str(uuid.uuid4())
    
    # 保存上传的文件
    input_path = os.path.join(TEMP_DIR, f"{temp_id}_input.mp4")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 使用SubtitleDetect进行字幕检测
        detector = SubtitleDetect(input_path)
        subtitle_frames = detector.find_subtitle_frame_no()
        
        # 统计字幕区域出现频率
        area_count = {}
        for frame_no, areas in subtitle_frames.items():
            for area in areas:
                area_str = str(area)
                area_count[area_str] = area_count.get(area_str, 0) + 1
        
        # 获取最常见的字幕区域
        common_areas = sorted(area_count.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 清理临时文件
        os.remove(input_path)
        
        return {
            "subtitle_detected": len(subtitle_frames) > 0,
            "total_frames_with_subtitles": len(subtitle_frames),
            "common_subtitle_areas": [
                {"area": area, "count": count} for area, count in common_areas
            ]
        }
    except Exception as e:
        # 清理临时文件
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

def process_video_task(task_id: str, input_path: str, model: str, subtitle_area: tuple = None):
    """
    后台处理视频的任务函数
    """
    try:
        # 更新任务状态
        tasks[task_id]["progress"] = 10
        
        # 根据选择的模型设置配置
        from backend import config
        
        # 调试打印：看看前端传过来的字符串是什么
        print(f"DEBUG_API: 接收到的 model 字符串 = '{model}'")
        
        target_mode = None
        if model == "sttn":
            target_mode = config.InpaintMode.STTN
        elif model == "lama":
            target_mode = config.InpaintMode.LAMA
        elif model == "propainter":
            target_mode = config.InpaintMode.PROPAINTER
        
        # 调试打印：看看转换后的 target_mode 是什么
        print(f"DEBUG_API: 转换后的 target_mode = {target_mode}")

        # 实例化
        subtitle_remover = SubtitleRemover(input_path, sub_area=subtitle_area, mode=target_mode)
        
        # 重写进度更新方法以更新任务状态
        original_update_progress = subtitle_remover.update_progress
        
        # 2. 定义包装函数
        # 关键点：这里不要写 self 参数，也不要用 __get__ 绑定
        # 我们直接把这个函数赋值给实例，它就会像普通函数一样被调用
        def update_progress_wrapper(tbar, increment=1):
            # 调用原始方法：这里只需要传 tbar 和 increment
            # Python 会自动处理 original_update_progress 里的 self
            original_update_progress(tbar, increment)
            
            # 进度计算逻辑：
            # 直接使用外部变量 subtitle_remover (闭包特性)，而不是通过 self 访问
            try:
                if subtitle_remover.frame_count > 0:
                    # 计算百分比
                    p_val = int((subtitle_remover.progress_total / subtitle_remover.frame_count) * 80 + 10)
                    tasks[task_id]["progress"] = min(p_val, 90)
            except Exception:
                pass # 防止进度计算出错影响主程序

        # 3. 直接赋值替换 (这是最稳妥的Monkey Patch方式)
        # 这样调用 subtitle_remover.update_progress(...) 时，就是直接调用上面的 wrapper 函数
        subtitle_remover.update_progress = update_progress_wrapper
        
        # 执行字幕去除
        subtitle_remover.run()
        
        # 创建输出路径
        output_path = os.path.join(TEMP_DIR, f"{task_id}_output.mp4")
        
        # 移动生成的文件到指定路径
        if hasattr(subtitle_remover, 'video_out_name') and os.path.exists(subtitle_remover.video_out_name):
            shutil.move(subtitle_remover.video_out_name, output_path)
        
        # 恢复原配置
        #config.MODE = old_mode
        
        # 更新任务状态
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["output_path"] = output_path
        
    except Exception as e:
        # 更新任务状态为失败
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"Task {task_id} failed: {str(e)}")
    finally:
        # 清理输入文件
        if os.path.exists(input_path):
            os.remove(input_path)

# 启动服务时的处理
@app.on_event("startup")
async def startup_event():
    print("FastAPI服务已启动")
    print(f"API文档地址: http://localhost:8000/docs")
    print(f"当前设备: {device}")
    print(f"ONNX执行提供者: {ONNX_PROVIDERS}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)