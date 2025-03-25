# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlpackage          # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
                              yolov8n.mnn                # MNN
                              yolov8n_ncnn_model         # NCNN
"""
import copy
import platform
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source, load_inference_source_multimodal    # TODO：多模态
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict): Dictionary of {save_path: video_writer, ...} writer for saving video output.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        # TODO: 增加红外图像
        self.plotted_img2 = None
        self.results2 = None
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    def preprocess_multimodal_v1(self, im, im2, extrinsics):
        """
        Preprocesses multimodal input images before inference.

        Args:
            im (torch.Tensor): RGB image tensor.
            im2 (torch.Tensor): Infrared image tensor.
        """
        not_tensor, not_tensor2 = not isinstance(im, torch.Tensor), not isinstance(im2, torch.Tensor)
        if not_tensor or not_tensor2:
            # 保存原始尺寸用于后处理
            self.orig_shape = {'rgb': im[0].shape[:2], 'ir': im2[0].shape[:2]}

            # 计算缩放比例
            self.scale_factors = {
                'rgb': (self.imgsz[0] / im[0].shape[0], self.imgsz[1] / im[0].shape[1]),
                'ir': (self.imgsz[0] / im2[0].shape[0], self.imgsz[1] / im2[0].shape[1])
            }

            # 使用letterbox进行预处理
            im = np.stack(self.pre_transform(im))
            im2 = np.stack(self.pre_transform(im2))
            
            # BGR to RGB, BHWC to BCHW
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im2 = im2[..., ::-1].transpose((0, 3, 1, 2))
            
            im = np.ascontiguousarray(im)
            im2 = np.ascontiguousarray(im2)
            
            im = torch.from_numpy(im)
            im2 = torch.from_numpy(im2)
            
        im = im.to(self.device)
        im2 = im2.to(self.device)
        
        im = im.half() if self.model.fp16 else im.float()
        im2 = im2.half() if self.model.fp16 else im2.float()
        
        if not_tensor:
            im /= 255
        if not_tensor2:
            im2 /= 255

        return im, im2
    
    def preprocess_multimodal(self, im, im2, extrinsics=None):
        """
        预处理多模态输入图像和对应的单应性矩阵。
        
        Args:
            im (torch.Tensor | np.ndarray): RGB图像 [B, H, W, C]
            im2 (torch.Tensor | np.ndarray): 红外图像 [B, H, W, C]
            extrinsics (torch.Tensor | np.ndarray, optional): 单应性矩阵 [B, 3, 3]
        
        Returns:
            tuple: (处理后的RGB图像, 处理后的红外图像, 更新后的单应性矩阵)
        """
        not_tensor = not isinstance(im, torch.Tensor)
        not_tensor2 = not isinstance(im2, torch.Tensor)
        
        if not_tensor or not_tensor2:
            batch_size = len(im)
            
            # 首先将extrinsics转换为torch.Tensor并移动到正确的设备
            if extrinsics is not None:
                if not isinstance(extrinsics, torch.Tensor):
                    extrinsics = torch.from_numpy(extrinsics)
                extrinsics = extrinsics.to(self.device)
                dtype = extrinsics.dtype
            else:
                dtype = torch.float32

            # 1. 保存原始尺寸
            self.orig_shapes = {
                'rgb': [im[i].shape[:2] for i in range(batch_size)],
                'ir': [im2[i].shape[:2] for i in range(batch_size)]
            }

            # 2. 计算缩放变换矩阵
            resize_matrices = []
            for i in range(batch_size):
                # RGB图像缩放矩阵
                rgb_h, rgb_w = self.orig_shapes['rgb'][i]
                scale_rgb_h = self.imgsz[0] / rgb_h
                scale_rgb_w = self.imgsz[1] / rgb_w
                
                # IR图像缩放矩阵
                ir_h, ir_w = self.orig_shapes['ir'][i]
                scale_ir_h = self.imgsz[0] / ir_h
                scale_ir_w = self.imgsz[1] / ir_w
                
                # 构建缩放变换矩阵
                S_rgb = torch.eye(3, device=self.device, dtype=dtype)
                S_rgb[0, 0] = scale_rgb_w  # x方向缩放
                S_rgb[1, 1] = scale_rgb_h  # y方向缩放
                
                S_ir = torch.eye(3, device=self.device, dtype=dtype)
                S_ir[0, 0] = scale_ir_w    # x方向缩放
                S_ir[1, 1] = scale_ir_h    # y方向缩放
                
                resize_matrices.append((S_rgb, S_ir))

            # 3. letterbox处理和padding变换矩阵计算
            padding_matrices = []
            for i in range(batch_size):
                # RGB图像letterbox
                rgb_img = im[i]
                rgb_h, rgb_w = rgb_img.shape[:2]
                r_rgb = min(self.imgsz[0]/rgb_h, self.imgsz[1]/rgb_w)
                new_unpad_rgb = int(round(rgb_w * r_rgb)), int(round(rgb_h * r_rgb))
                dw_rgb, dh_rgb = self.imgsz[1] - new_unpad_rgb[0], self.imgsz[0] - new_unpad_rgb[1]
                dw_rgb /= 2
                dh_rgb /= 2
                
                # IR图像letterbox
                ir_img = im2[i]
                ir_h, ir_w = ir_img.shape[:2]
                r_ir = min(self.imgsz[0]/ir_h, self.imgsz[1]/ir_w)
                new_unpad_ir = int(round(ir_w * r_ir)), int(round(ir_h * r_ir))
                dw_ir, dh_ir = self.imgsz[1] - new_unpad_ir[0], self.imgsz[0] - new_unpad_ir[1]
                dw_ir /= 2
                dh_ir /= 2
                
                # 构建padding变换矩阵
                P_rgb = torch.eye(3, device=self.device, dtype=dtype)
                P_rgb[0, 2] = dw_rgb  # x方向平移
                P_rgb[1, 2] = dh_rgb  # y方向平移
                
                P_ir = torch.eye(3, device=self.device, dtype=dtype)
                P_ir[0, 2] = dw_ir  # x方向平移
                P_ir[1, 2] = dh_ir  # y方向平移
                
                padding_matrices.append((P_rgb, P_ir))

            # 4. 应用所有变换
            if extrinsics is not None:
                if not isinstance(extrinsics, torch.Tensor):
                    extrinsics = torch.from_numpy(extrinsics).to(self.device)
                
                # 为每个样本更新映射矩阵
                transformed_H = []
                for i in range(batch_size):
                    S_rgb, S_ir = resize_matrices[i]
                    P_rgb, P_ir = padding_matrices[i]
                    
                    # 计算新的映射矩阵
                    # 新IR -> 原始RGB的变换顺序：
                    # 1. 移除IR的padding: P_ir^(-1)
                    # 2. 移除IR的缩放: S_ir^(-1)
                    # 3. 应用原始映射: H
                    # 4. 应用RGB缩放: S_rgb
                    # 5. 应用RGB padding: P_rgb
                    
                    P_ir_inv = torch.inverse(P_ir)
                    S_ir_inv = torch.inverse(S_ir)
                    
                    current_H = extrinsics[i]  # 原始映射
                    current_H = torch.mm(P_rgb, torch.mm(S_rgb, torch.mm(current_H, 
                                       torch.mm(S_ir_inv, P_ir_inv))))
                    
                    transformed_H.append(current_H)
                
                extrinsics = torch.stack(transformed_H)

            # 5. 应用letterbox变换到图像
            letterbox = LetterBox(
                self.imgsz,
                auto=False,  # 关闭自动模式以确保一致的处理
                stride=self.model.stride
            )
            im = np.stack([letterbox(image=x) for x in im])
            im2 = np.stack([letterbox(image=x) for x in im2])

            # 6. 图像格式转换
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im2 = im2[..., ::-1].transpose((0, 3, 1, 2))
            
            im = np.ascontiguousarray(im)
            im2 = np.ascontiguousarray(im2)
            
            # 转换为tensor并移动到正确的设备
            im = torch.from_numpy(im)
            im2 = torch.from_numpy(im2)

        # 确保所有张量都在正确的设备上
        im = im.to(self.device)
        im2 = im2.to(self.device)
        if extrinsics is not None:
            extrinsics = extrinsics.to(self.device)

        # 数据类型转换
        im = im.half() if self.model.fp16 else im.float()
        im2 = im2.half() if self.model.fp16 else im2.float()
        if extrinsics is not None:
            extrinsics = extrinsics.half() if self.model.fp16 else extrinsics.float()
        
        # 归一化
        if not_tensor:
            im /= 255
        if not_tensor2:
            im2 /= 255

        return im, im2, extrinsics

    def letterbox(self, im, stride=32):
        """
        对图像进行letterbox处理，返回处理后的图像和相关参数
        """
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
        
        return im, {'pad': (left, top), 'ratio': r}

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def inference_multimodal(self, im, im2, extrinsics, *args, **kwargs):
        """Runs inference on multimodal input images using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model.forward_multimodal(im, im2, extrinsics, augment=self.args.augment, visualize=self.args.visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]
    
    def letterbox_and_update_extrinsics(self, im, im2, extrinsics=None):
        """
        对图像进行letterbox操作并更新单应性矩阵
        
        Args:
            im (np.ndarray): RGB图像 [B, H, W, C]
            im2 (np.ndarray): IR图像 [B, H, W, C]
            extrinsics (torch.Tensor, optional): 单应性矩阵 [B, 3, 3]
        """
        # 1. 创建letterbox对象
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        
        # 2. 获取批次大小
        batch_size = len(im)
        
        # 3. 为每个样本计算padding值
        pad_values = []
        for i in range(batch_size):
            # 获取当前样本的原始尺寸
            shape = im[i].shape[:2]    # RGB图像尺寸
            shape2 = im2[i].shape[:2]  # IR图像尺寸
            
            # 计算缩放比例
            r = min(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1])
            r2 = min(self.imgsz[0] / shape2[0], self.imgsz[1] / shape2[1])
            
            # 计算新的未填充尺寸
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            new_unpad2 = int(round(shape2[1] * r2)), int(round(shape2[0] * r2))
            
            # 计算padding
            dw, dh = self.imgsz[1] - new_unpad[0], self.imgsz[0] - new_unpad[1]
            dw2, dh2 = self.imgsz[1] - new_unpad2[0], self.imgsz[0] - new_unpad2[1]
            
            # 存储当前样本的padding值
            pad_values.append({
                'rgb': (dw // 2, dh // 2),
                'ir': (dw2 // 2, dh2 // 2)
            })
        
        # 4. 进行letterbox处理
        im = np.stack([letterbox(image=x) for x in im])
        im2 = np.stack([letterbox(image=x) for x in im2])

        # 5. 更新单应性矩阵以反映padding
        if extrinsics is not None:
            if not isinstance(extrinsics, torch.Tensor):
                extrinsics = torch.from_numpy(extrinsics)
                
            # 为每个样本创建padding变换矩阵 [B, 3, 3]
            H_pad_rgb = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(extrinsics.device)
            H_pad_ir = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(extrinsics.device)
            
            # 为每个样本设置padding值
            for i in range(batch_size):
                pad_rgb_x, pad_rgb_y = pad_values[i]['rgb']
                pad_ir_x, pad_ir_y = pad_values[i]['ir']
                
                # RGB图像的padding变换矩阵的逆矩阵
                H_pad_rgb[i, 0, 2] = -pad_rgb_x  # x方向反向平移
                H_pad_rgb[i, 1, 2] = -pad_rgb_y  # y方向反向平移
                
                # IR图像的padding变换矩阵
                H_pad_ir[i, 0, 2] = pad_ir_x  # x方向平移
                H_pad_ir[i, 1, 2] = pad_ir_y  # y方向平移
            
            # 更新映射关系：新的IR -> 原始IR -> 原始RGB -> 新的RGB
            # 使用torch.bmm进行批次矩阵乘法
            extrinsics = torch.bmm(H_pad_rgb, extrinsics)  # [B, 3, 3]
            extrinsics = torch.bmm(extrinsics, H_pad_ir)   # [B, 3, 3]
        
        return im, im2, extrinsics

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if self.args.task == 'multimodal':  # 多模态任务
            if stream:
                return self.stream_inference_multimodal(source, model, *args, **kwargs)
            else:
                return list(self.stream_inference_multimodal(source, model, *args, **kwargs))  # merge list of Result into one
        else :
            if stream:
                return self.stream_inference(source, model, *args, **kwargs)
            else:
                return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        if self.args.task == 'multimodal':  # 多模态任务
            gen = self.stream_inference_multimodal(source, model)
        else:
            gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )

        if self.args.task == 'multimodal':  # 多模态任务
            self.dataset = load_inference_source_multimodal(
                source=source,
                batch=self.args.batch,
                vid_stride=self.args.vid_stride,
                buffer=self.args.stream_buffer,
            )

            self.source_type = self.dataset[0].source_type
            if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset[0]) > 1000  # many images
            or any(getattr(self.dataset[0], "video_flag", [False]))
        ):  # videos
                LOGGER.warning(STREAM_WARNING)
        else:
            self.dataset = load_inference_source(
                source=source,
                batch=self.args.batch,
                vid_stride=self.args.vid_stride,
                buffer=self.args.stream_buffer,
            )

            self.source_type = self.dataset.source_type
            if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def stream_inference_multimodal(self, source=None, model=None, *args, **kwargs):     # TODO：多模态流式推理
        """Streams real-time inference on multimodal camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup_multimodal(imgsz=(1 if self.model.pt or self.model.triton else self.dataset[0].bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch1, batch2, extrinsics in zip(self.dataset[0], self.dataset[1], self.dataset[2]):
                self.run_callbacks("on_predict_batch_start")
                self.batch = batch1, batch2, extrinsics
                paths, im0s, s = batch1
                paths2, im0s2, s2 = batch2

                # if extrinsincs is not None:
                #     extrinsics = batch1.extrinsics
                # else:
                #     extrinsics = None

                # Preprocess
                with profilers[0]:
                    im, im2, extrinsics = self.preprocess_multimodal(im0s, im0s2, extrinsics)

                # Inference
                with profilers[1]:
                    preds = self.inference_multimodal(im, im2, extrinsics, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    preds2 = copy.deepcopy(preds)
                    self.results = self.postprocess(preds, im, im0s, 0)
                    self.results2 = self.postprocess(preds2, im2, im0s2, 1)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    self.results2[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results_multimodal(i, Path(paths[i]), im, s, 'rgb')
                        s2[i] += self.write_results_multimodal(i, Path(paths2[i]), im2, s2, 'ir')
                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))
                    LOGGER.info("\n".join(s2))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results
                yield from self.results2
        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
        

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def write_results_multimodal(self, i, p, im, s, img_type='rgb'):
        """Write inference results to a file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined
        
        # 根据图像类型设置不同的路径和结果
        if img_type == 'rgb':
            self.txt_path = self.save_dir / "labels_vi" / (p.stem + ("" if self.dataset[0].mode == "image" else f"_{frame}"))
            result = self.results[i]
        else:
            self.txt_path = self.save_dir / "labels_ir" / (p.stem + ("" if self.dataset[1].mode == "image" else f"_{frame}"))
            result = self.results2[i]
            
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.show_boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels,
                'im_gpu': None if self.args.retina_masks else im[i]
            }
            
            if img_type == 'rgb':
                orig_shape = self.orig_shapes['rgb'][i]
                result.orig_shape = orig_shape  # 设置原始尺寸
                self.plotted_img = result.plot(**plot_args)
            else:
                orig_shape = self.orig_shapes['ir'][i]
                result.orig_shape = orig_shape  # 设置原始尺寸
                self.plotted_img2 = result.plot(**plot_args)

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show_multimodal(str(p), img_type)
        if self.args.save:
            # 根据图像类型设置不同的保存路径
            prefix = "visible" if img_type == "rgb" else "infrared"
            save_path = str(self.save_dir / f"{prefix}_{p.name}")
            self.save_predicted_images_multimodal(save_path, frame, img_type)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """Save video predictions as mp4 at specified path."""
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f"{save_path.split('.', 1)[0]}_frames/"
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # save to JPG for best support

    def save_predicted_images_multimodal(self, save_path="", frame=0, img_type='rgb'):  # TODO：多模态保存
        """Save video predictions as mp4 at specified path."""
        # 根据图像类型选择对应的plotted图像
        im = self.plotted_img if img_type == 'rgb' else self.plotted_img2

        # Save videos and streams
        if self.dataset[0].mode in {"stream", "video"} and self.dataset[1].mode in {"stream", "video"}:
            fps = self.dataset[0].fps if self.dataset[0].mode == "video" else 30
            frames_path = f"{save_path.split('.', 1)[0]}_frames/"
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # save to JPG for best support

    def show(self, p=""):
        """Display an image in a window using the OpenCV imshow function."""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 millisecond
    
    def show_multimodal(self, p="", img_type='rgb'):    # TODO：多模态显示
        """Display an image in a window using the OpenCV imshow function."""
        if img_type == 'rgb':
            im = self.plotted_img
        elif img_type == 'ir':
            im = self.plotted_img2
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset[0].mode == "image" else 1)  # 1 millisecond

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
