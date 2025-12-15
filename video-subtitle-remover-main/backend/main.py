import torch
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.tools.common_tools import is_video_or_image, is_image_file
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import multiprocessing
from shapely.geometry import Polygon
import time
from tqdm import tqdm


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_area=None):
        self.video_path = video_path
        self.sub_area = sub_area

    @cached_property
    def text_detector(self):
        from paddleocr import PaddleOCR
        import numpy as np
        import logging
        import os
        
        # 屏蔽日志
        logging.getLogger("ppocr").setLevel(logging.WARNING)

        # 检查模型路径
        if not os.path.exists(config.DET_MODEL_PATH):
            print(f"[Warning] 本地模型路径不存在: {config.DET_MODEL_PATH}，PaddleOCR 将尝试自动下载")
            model_dir = None
        else:
            print(f"[Info] 使用本地模型路径: {config.DET_MODEL_PATH}")
            model_dir = config.DET_MODEL_PATH

        # 初始化 PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            show_log=False,
            # 优先使用 GPU，如果没检测到 cuda 则自动回退 CPU
            use_gpu=True, 
            
            # 【关键修改】强制关闭 ONNX
            # 因为我们已经安装了 paddlepaddle-gpu，原生推理更稳定
            # 开启 ONNX 会导致 Protobuf 解析错误
            use_onnx=False,
            
            # 指定本地模型路径
            det_model_dir=model_dir
        )

        def predict_wrapper(img):
            try:
                # 核心调用
                result = ocr.ocr(img, rec=False, cls=False)
            except Exception as e:
                print(f"[Error] PaddleOCR detection error: {e}")
                result = None

            dt_boxes = []
            if result is not None:
                if len(result) > 0 and result[0] is not None:
                    dt_boxes = result[0]
            
            return np.array(dt_boxes), 0

        return predict_wrapper

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        [必须修改] 获取检测框的最小外包矩形，防止倾斜导致坐标错误
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                # 获取四个点的坐标
                x_list = [int(i[0][0]), int(i[1][0]), int(i[2][0]), int(i[3][0])]
                y_list = [int(i[0][1]), int(i[1][1]), int(i[2][1]), int(i[3][1])]
                
                # 使用 min/max 确保无论 OCR 返回的框是否倾斜，都能完整包裹住文字
                xmin = min(x_list)
                xmax = max(x_list)
                ymin = min(y_list)
                ymax = max(y_list)
                
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                if len(temp_list) > 0:
                    subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        # if config.UNITE_COORDINATES:
        #     subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(subtitle_frame_no_box_dict)
        #     if sub_remover is not None:
        #         try:
        #             # 当帧数大于1时，说明并非图片或单帧
        #             if sub_remover.frame_count > 1:
        #                 subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
        #                                                                           sub_remover.fps)
        #         except Exception:
        #             pass
        #     subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir


    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # 初始化输出区间列表
        expanded_intervals = []

        # 对每个原始区间进行扩展
        for interval in intervals:
            start, end = interval

            # 扩展至至少 'expand_size' 个单位，但不超过 'max_length' 个单位
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # 在保证包含原区间的前提下尽可能平分前后扩展量
            expand_start = max(start - expansion_amount // 2, 1)  # 确保起始点不小于1
            expand_end = end + expansion_amount // 2

            # 如果扩展后的区间超出了最大长度，进行调整
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # 对于单点的处理，需额外保证有至少 'expand_size' 长度
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # 检查与前一个区间是否有重叠并进行相应的合并
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # 添加扩展后的区间至结果列表
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        """
        expanded = []
        # 首先单独处理单点区间以扩展它们
        for start, end in intervals:
            if start == end:  # 单点区间
                # 扩展到接近的目标长度，但保证前后不重叠
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # 查找下一个区间的起始点
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # 确定新的扩展起点和终点
                calc_start = start - (target_length - 1) // 2
                new_start = max(calc_start, prev_end + 1, 1)  # 这里加了个 1
                #new_start = max(start - (target_length - 1) // 2, prev_end + 1)

                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # 如果新的扩展终点在起点前面，说明没有足够空间来进行扩展
                if new_end < new_start:
                    new_start, new_end = start, start  # 保持原样
                expanded.append((new_start, new_end))
            else:
                # 非单点区间直接保留，稍后处理任何可能的重叠
                expanded.append((start, end))
        # 排序以合并那些因扩展导致重叠的区间
        expanded.sort(key=lambda x: x[0])
        # 合并重叠的区间，但仅当它们之间真正重叠且小于目标长度时
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # 检查是否重叠
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 需要合并
                merged[-1] = (last_start, max(last_end, end))  # 合并区间
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 相邻区间也需要合并的场景
                merged[-1] = (last_start, end)
            else:
                # 如果没有重叠且都大于目标长度，则直接保留
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # 寻找面积最大文本框
            current_no = start_no
            # 查找当前区间矩形框最大面积
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # 取出每一个文本框坐标
                    xmin, xmax, ymin, ymax = coord
                    # 计算当前文本框坐标面积
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # 如果区间最大框列表为空，则当前面积为区间最大面积
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # 如果列表非空，判断当前文本框是与区间最大文本框在同一区域
                    else:
                        has_same_position = False
                        # 遍历每个区间最大文本框，判断当前文本框位置是否与区间最大文本框列表的某个文本框位于同一行且交叉
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # 如果高度差异不一样
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # 如果在同一行，则计算当前面积是不是最大
                                    # 判断面积大小，若当前面积更大，则将当前行的最大区域坐标点更新
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # 如果遍历了所有的区间最大文本框列表，发现是新的一行，则直接添加
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        将多个视频帧的文本区域坐标统一
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        添加额外的文本框，防止漏检
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        过滤错误的字幕区域
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, gui_mode=False, mode=None):
        # 调试打印：看看传入的 mode 到底是不是 None
        print(f"DEBUG: SubtitleRemover 初始化, 传入 mode={mode}")
        
        # 这里的逻辑必须是这样
        self.mode = mode if mode else config.MODE
        
        # 调试打印：看看最终确定的 self.mode 是什么
        print(f"DEBUG: 最终确定 self.mode={self.mode}")

        self.mode = mode if mode else config.MODE
        #importlib.reload(config)
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_area = sub_area
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        # 判断是否为图片
        self.is_picture = False
        if is_image_file(str(vd_path)):
            self.sub_area = None
            self.is_picture = True
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        if config.USE_DML:
            print('use DirectML for acceleration')
            if config.MODE != config.InpaintMode.STTN:
                print('Warning: DirectML acceleration is only available for STTN model. Falling back to CPU for other models.')
        for provider in config.ONNX_PROVIDERS:
            print(f"Detected execution provider: {provider}")


        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 预览帧
        self.preview_frame = None
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def propainter_mode(self, tbar):
        print('use propainter mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                print(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # print(f'No 1 Current index: {index}')
                    start_frame_no = index
                    print(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        print(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            # === 修改开始：单帧膨胀逻辑 ===
                            expanded_areas = []
                            pad = config.SUBTITLE_AREA_DEVIATION_PIXEL
                            for area in sub_list[index]:
                                xmin, xmax, ymin, ymax = area
                                ymin = max(0, ymin - pad)
                                ymax = min(self.frame_height, ymax + pad)
                                xmin = max(0, xmin - pad)
                                xmax = min(self.frame_width, xmax + pad)
                                expanded_areas.append((xmin, xmax, ymin, ymax))
                            single_mask = create_mask(self.mask_size, expanded_areas)
                            # === 修改结束 ===
                            
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            
                            # === 修改开始：批次Mask膨胀逻辑 ===
                            expanded_areas = []
                            pad = config.SUBTITLE_AREA_DEVIATION_PIXEL
                            # ProPainter 逻辑比较特殊，它这里默认是用 start_frame_no 的 mask 代表这一整段
                            # 为了保证覆盖，我们同样进行膨胀
                            if start_frame_no in sub_list:
                                for area in sub_list[start_frame_no]:
                                    xmin, xmax, ymin, ymax = area
                                    ymin = max(0, ymin - pad)
                                    ymax = min(self.frame_height, ymax + pad)
                                    xmin = max(0, xmin - pad)
                                    xmax = min(self.frame_width, xmax + pad)
                                    expanded_areas.append((xmin, xmax, ymin, ymax))
                            
                            mask = create_mask(self.mask_size, expanded_areas)
                            # === 修改结束 ===

                            for batch in batch_generator(temp_frames, config.PROPAINTER_MAX_LOAD_NUM):
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    # 单帧兜底逻辑也需要膨胀
                                    single_mask = mask # 直接复用上面生成的 mask 即可
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = self.video_inpaint.inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        if self.gui_mode:
                                            self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                                self.update_progress(tbar, increment=len(batch))

    def sttn_mode_with_no_detection(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
            print(f"DEBUG: Original sub_area: {ymin}, {ymax}, {xmin}, {xmax}")
            
            # 2. 【关键步骤】手动应用膨胀逻辑
            # 向四周扩大选区，彻底盖住黄色光晕
            ymin = max(0, ymin - pad)
            ymax = min(self.frame_height, ymax + pad)
            xmin = max(0, xmin - pad)
            xmax = min(self.frame_width, xmax + pad)
            
            print(f"DEBUG: Expanded sub_area: {ymin}, {ymax}, {xmin}, {xmax}")
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def sttn_mode(self, tbar):
        """
        STTN 模式（最终稳定版）
        核心思想：
        - union_mask：只用于计算 STTN 的 inpaint_area
        - mask_getter：决定每一帧是否真的使用 STTN 结果
        """
        print("\n" + "=" * 50)
        print("【STTN】Per-frame Mask Inpainting Mode")
        print("=" * 50 + "\n")

        # 1. 字幕检测
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)

        if not sub_list:
            print("[STTN] No subtitles detected, copy original video.")
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                self.video_writer.write(frame)
                self.update_progress(tbar, 1)
            return

        # 2. 自适应 pad（720p / 1080p 通用）
        pad_y = max(10, min(28, int(self.frame_height * 0.02)))
        pad_x = max(14, min(42, int(self.frame_height * 0.03)))

        # 3. 构建时域稳定的 per-frame mask dict
        frames = sorted(sub_list.keys())
        final_mask_dict = {}

        current_group = [frames[0]]
        groups = []

        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > 5:
                groups.append(current_group)
                current_group = []
            current_group.append(frames[i])
        groups.append(current_group)

        for group in groups:
            all_boxes = []
            for f in group:
                all_boxes.extend(sub_list[f])

            if not all_boxes:
                continue

            # 按 Y 中心聚类（区分不同行字幕）
            clusters = []
            for box in all_boxes:
                cy = (box[2] + box[3]) / 2
                for c in clusters:
                    if abs(c["cy"] - cy) < 100:
                        c["boxes"].append(box)
                        break
                else:
                    clusters.append({"cy": cy, "boxes": [box]})

            stable_boxes = []
            for c in clusters:
                xs = [b[0] for b in c["boxes"]] + [b[1] for b in c["boxes"]]
                ys = [b[2] for b in c["boxes"]] + [b[3] for b in c["boxes"]]

                xmin = max(0, min(xs) - pad_x)
                xmax = min(self.frame_width, max(xs) + pad_x)
                ymin = max(0, min(ys) - pad_y)
                ymax = min(self.frame_height, max(ys) + pad_y)

                stable_boxes.append((xmin, xmax, ymin, ymax))

            for f in group:
                final_mask_dict[f] = stable_boxes

        # 4. union mask（只用于 inpaint_area）
        all_static_boxes = []
        for boxes in final_mask_dict.values():
            all_static_boxes.extend(boxes)

        union_mask = create_mask(self.mask_size, all_static_boxes)

        # 5. per-frame mask getter（决定是否修）
        def mask_getter(frame_no: int):
            boxes = final_mask_dict.get(frame_no, [])
            return create_mask(self.mask_size, boxes)

        # 6. 启动 STTN
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(
            input_mask=union_mask,
            input_sub_remover=self,
            tbar=tbar,
            mask_getter=mask_getter
        )


    # def sttn_mode(self, tbar):
    #     """
    #     STTN模式主入口 - [最终稳定版]
    #     修复策略：时域平滑 + 激进膨胀 + 强制稳定Mask
    #     """
    #     print("\n" + "="*50)
    #     print("【执行】时域稳定修复模式 (Temporal Stable Inpainting)")
    #     print("="*50 + "\n")

    #     # if config.STTN_SKIP_DETECTION:
    #     #     self.sttn_mode_with_no_detection(tbar)
    #     #     return

    #     # 1. 检测字幕
    #     print('Step 1: 扫描全视频字幕...')
    #     sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        
    #     if not sub_list:
    #         print("警告：未检测到任何字幕！")
    #         while True:
    #             ret, frame = self.video_cap.read()
    #             if not ret: break
    #             self.video_writer.write(frame)
    #             self.update_progress(tbar, increment=1)
    #         return

    #     # 2. 【关键步骤】时域平滑与合并
    #     # 目的是：算出每一行字幕在出现期间的“最大范围”，防止Mask忽大忽小导致泄露
    #     print("Step 2: 计算全局最大遮罩范围...")
        
    #     # 参数设置
    #     # 激进的膨胀参数，专门针对带光晕的字幕
    #     pad_x = 50  # 左右多扩40像素
    #     pad_y = 40  # 上下多扩30像素
        
    #     # 将 sub_list 转换为列表以便处理
    #     frames_with_subs = sorted(list(sub_list.keys()))
    #     if not frames_with_subs:
    #         return

    #     # 简单的聚类算法，将连续的帧视为同一组字幕
    #     groups = []
    #     if frames_with_subs:
    #         current_group = [frames_with_subs[0]]
    #         for i in range(1, len(frames_with_subs)):
    #             # 如果帧号中断超过 5 帧，视为下一句字幕
    #             if frames_with_subs[i] - frames_with_subs[i-1] > 5:
    #                 groups.append(current_group)
    #                 current_group = []
    #             current_group.append(frames_with_subs[i])
    #         groups.append(current_group)

    #     # 构建最终的 Mask 字典
    #     # key: frame_no, value: list of (xmin, xmax, ymin, ymax)
    #     final_mask_dict = {}

    #     for group in groups:
    #         if not group: continue
            
    #         # 1. 收集这一组时间段内，所有检测到的框
    #         all_boxes_in_group = []
    #         for f in group:
    #             all_boxes_in_group.extend(sub_list[f])
            
    #         # 2. 分离这一组里的不同区域（比如顶部标题和底部字幕）
    #         # 我们通过 Y 轴坐标来聚类
    #         y_clusters = [] 
    #         for box in all_boxes_in_group:
    #             # box: xmin, xmax, ymin, ymax
    #             cy = (box[2] + box[3]) / 2  # 中心点Y
    #             found_cluster = False
    #             for cluster in y_clusters:
    #                 # 如果中心点距离在 100 像素内，视为同一行
    #                 if abs(cluster['center_y'] - cy) < 100:
    #                     cluster['boxes'].append(box)
    #                     # 更新聚类中心
    #                     cluster['center_y'] = (cluster['center_y'] * cluster['count'] + cy) / (cluster['count'] + 1)
    #                     cluster['count'] += 1
    #                     found_cluster = True
    #                     break
    #             if not found_cluster:
    #                 y_clusters.append({'center_y': cy, 'boxes': [box], 'count': 1})
            
    #         # 3. 对每一行字幕，计算“最大并集框” (Union Box)
    #         # 这就是消除“鬼影”的核心：只要有一帧检测到了完整的字，所有帧都用那个最大的框
    #         final_boxes_for_this_group = []
    #         for cluster in y_clusters:
    #             c_boxes = cluster['boxes']
    #             # 获取这一行在所有帧里出现过的最左、最右、最上、最下坐标
    #             u_xmin = min([b[0] for b in c_boxes])
    #             u_xmax = max([b[1] for b in c_boxes])
    #             u_ymin = min([b[2] for b in c_boxes])
    #             u_ymax = max([b[3] for b in c_boxes])
                
    #             # 应用激进膨胀
    #             f_xmin = max(0, u_xmin - pad_x)
    #             f_xmax = min(self.frame_width, u_xmax + pad_x)
    #             f_ymin = max(0, u_ymin - pad_y)
    #             f_ymax = min(self.frame_height, u_ymax + pad_y)
                
    #             final_boxes_for_this_group.append((f_xmin, f_xmax, f_ymin, f_ymax))
            
    #         # 4. 将计算好的固定Mask应用到这一组的所有帧
    #         for f in group:
    #             final_mask_dict[f] = final_boxes_for_this_group

    #     # 3. 生成 Mask 序列
    #     # STTN 需要每一帧的 Mask，这里我们需要把 dict 转换成 STTNVideoInpaint 能识别的格式
    #     # 但为了复用现有逻辑，我们这里生成一个全局静态 Mask 列表给 create_mask
    #     # 注意：STTNInpaint 内部会根据帧号去读，但这里我们简化处理，
    #     # 我们需要重写一个 generator 或者直接传 dict 进去比较麻烦。
    #     # 最简单的方法：修改 STTNVideoInpaint 里的逻辑，或者在这里生成一个 mask list
        
    #     # 修正：由于 STTNVideoInpaint 默认逻辑比较死板，我们这里生成一个覆盖所有帧的 mask_list
    #     # 这会导致性能下降（每帧都算），但效果最好。
        
    #     # 既然不能直接改 STTN 内部循环，我们用一种 Hack 的方法：
    #     # 我们把 mask 画在每一帧上？不行。
    #     # 我们构建一个 mask_generator。
        
    #     print("Step 3: 启动 STTN 引擎...")
        
    #     # 定义一个内部回调，用于获取每一帧的 mask
    #     def get_mask_for_frame(frame_idx):
    #         # +1 因为 frame_no 通常从1开始
    #         current_no = frame_idx + 1
    #         if current_no in final_mask_dict:
    #             return create_mask(self.mask_size, final_mask_dict[current_no])
    #         else:
    #             # 如果这一帧没字幕，返回全黑
    #             return create_mask(self.mask_size, [])

    #     # 实例化并运行
    #     sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        
    #     # 我们需要一点点 Hack，把处理好的 mask 逻辑注入进去
    #     # 或者我们直接生成一个巨大的 Mask 视频？太慢。
    #     # 让我们使用最原始的方法：传递一个 mask_path 或者 mask_generator
    #     # 鉴于你的代码结构，最稳妥的方式是把 final_mask_dict 传给 STTNVideoInpaint
    #     # 但我没有 STTNVideoInpaint 的源码权限，所以我只能假设它接受 input_mask
        
    #     # 如果 STTNVideoInpaint 只接受一张静态 mask (input_mask 参数)：
    #     # 那我们就必须把 final_mask_dict 里所有的框都画在一张图上
    #     # 缺点：即使字幕消失了，Mask还在，会修复空气，导致模糊。
    #     # 优点：绝对不会有残留。
        
    #     # === 方案 A：生成一张包含所有时间段字幕的静态大 Mask (最稳，但也最暴力) ===
    #     all_static_boxes = []
    #     for f, boxes in final_mask_dict.items():
    #         for b in boxes:
    #             if b not in all_static_boxes:
    #                 all_static_boxes.append(b)
        
    #     # 过滤一下重叠的
    #     mask = create_mask(self.mask_size, all_static_boxes)
        
    #     # 诊断图片
    #     debug_mask_path = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_stable_mask.png')
    #     cv2.imwrite(debug_mask_path, mask)
    #     print(f"【诊断】Mask已保存至: {debug_mask_path} (此处应包含所有出现过的字幕区域)")

    #     # sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)
    #     union_mask = create_mask(self.mask_size, all_static_boxes)

    #     def mask_getter(frame_no: int):
    #         # frame_no 按你 main.py 的逻辑是从 1 开始
    #         boxes = final_mask_dict.get(frame_no, [])
    #         return create_mask(self.mask_size, boxes)

    #     sttn_video_inpaint(
    #         input_mask=union_mask,
    #         input_sub_remover=self,
    #         tbar=tbar,
    #         mask_getter=mask_getter
    #     )

    def lama_mode(self, tbar):
        print('use lama mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        index = 0
        print('[Processing] start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list.keys():
                #mask = create_mask(self.mask_size, sub_list[index])
                # ============= 修改开始：加入Mask膨胀逻辑 =============
                expanded_areas = []
                for area in sub_list[index]:
                    xmin, xmax, ymin, ymax = area
                    pad = config.SUBTITLE_AREA_DEVIATION_PIXEL
                    # 强制扩大选区
                    ymin = max(0, ymin - pad)
                    ymax = min(self.frame_height, ymax + pad)
                    xmin = max(0, xmin - pad)
                    xmax = min(self.frame_width, xmax + pad)
                    
                    expanded_areas.append((xmin, xmax, ymin, ymax))
                
                # 使用扩大后的区域生成 Mask
                mask = create_mask(self.mask_size, expanded_areas)
                # ============= 修改结束 =============
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            tbar.update(1)
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover

    def run(self):
        # 记录开始时间
        start_time = time.time()
        # 重置进度条
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.lama_inpaint = LamaInpaint()
            original_frame = cv2.imread(self.video_path)
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list[1])
                inpainted_frame = self.lama_inpaint(original_frame, mask)
            else:
                inpainted_frame = original_frame
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
            tbar.update(1)
            self.progress_total = 100
        else:
            current_mode_str = self.mode.value if hasattr(self.mode, 'value') else str(self.mode)
            
            # 打印最终用来判断的字符串，确保万无一失
            print(f"DEBUG: 正在进行模式分发，当前模式值='{current_mode_str}'")

            # 2. 直接与字符串常量进行比较 (Robust!)
            if current_mode_str == 'propainter':
                print("DEBUG: 命中 Propainter 模式")
                self.propainter_mode(tbar)
            elif current_mode_str == 'sttn':
                print("DEBUG: 命中 STTN 模式")
                self.sttn_mode(tbar)
            else:
                print("DEBUG: 未命中特定模式，进入 Lama/Default 模式")
                self.lama_mode(tbar)

        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('fail to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [config.FFMPEG_PATH,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "libx264" if config.USE_H264 else "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_name]
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
                except Exception:
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() in ['Windows']:
                        pass
                    else:
                        print(f'failed to delete temp file {temp.name}')
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 1. 提示用户输入视频路径
    video_path = input(f"Please input video or image file path: ").strip()
    # 判断视频路径是不是一个目录，是目录的化，批量处理改目录下的所有视频文件
    # 2. 按以下顺序传入字幕区域
    # sub_area = (ymin, ymax, xmin, xmax)
    # 3. 新建字幕提取对象
    if is_video_or_image(video_path):
        sd = SubtitleRemover(video_path, sub_area=None)
        sd.run()
    else:
        print(f'Invalid video path: {video_path}')
