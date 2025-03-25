# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    MultiModalTransformer,
    DEA,
    FMDEA,
    EnhancedFMDEA
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

# TODO:多模态目标检测
class MultiModalDetectionModel(BaseModel):
    """YOLOv11 多模态融合检测模型."""

    def __init__(self, cfg="yolo11n-EnhancedFMDEA.yaml", ch=3, ch2=3, nc=None, verbose=True):
        """
        初始化 YOLOv11 多模态融合检测模型。

        Args:
            cfg (str | dict): 模型配置文件路径或配置字典。
            ch (tuple): 输入通道数，格式为 (可见光通道数, 红外通道数)。
            nc (int, optional): 类别数量。如果提供，将覆盖配置文件中的类别数。
            verbose (bool): 是否输出详细信息。
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 载入配置
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` 模块已弃用，使用 nn.Identity 代替。"
                "请删除本地 *.pt 文件并重新下载最新的模型权重。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        self.ch_visible, self.ch_thermal = ch, ch2  # 可见光和红外输入通道
        self.yaml["ch_visible"] = self.ch_visible
        self.yaml["ch_thermal"] = self.ch_thermal
        if nc and nc != self.yaml.get("nc", None):
            LOGGER.info(f"覆盖 model.yaml 中的 nc={self.yaml.get('nc', '未定义')} 为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖类别数

        # 解析模型
        self.model, self.save = parse_model_fusion(deepcopy(self.yaml), ch=self.ch_visible, ch2=self.ch_thermal, verbose=verbose)  # 解析模型
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 类别名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 构建步幅
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包含所有 Detect 子类如 Segment, Pose, OBB, WorldDetect
            s = 256  # 2x 最小步幅
            m.inplace = self.inplace

            def _forward(x_vis, x_therm, extrinsics):
                """通过模型前向传播，处理不同的 Detect 子类类型。"""
                if self.end2end:
                    return self.forward(x_vis, x_therm, extrinsics)["one2many"]
                return self.forward(x_vis, x_therm, extrinsics)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x_vis, x_therm, extrinsics)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, self.ch_visible, s, s), 
                                                                       torch.zeros(1, self.ch_thermal, s, s),
                                                                       torch.eye(3).expand(1, 3, 3))])  # 前向传播

            self.stride = m.stride
            m.bias_init()  # 仅运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步幅，例如 RTDETR

        # 初始化权重与偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def forward(self, x_vis, x_therm, extrinsics=None, *args, **kwargs):
        """
        执行模型的前向传播。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            extrinsics (torch.Tensor): 外参矩阵。
            *args: 额外的位置参数。
            **kwargs: 额外的关键字参数。

        Returns:
            torch.Tensor: 模型的输出。
        """
        if isinstance(x_vis, dict) and isinstance(x_therm, dict):
            return self.loss(x_vis, x_therm, extrinsics, *args, **kwargs)
        return self.predict(x_vis, x_therm, extrinsics, *args, **kwargs)
    
    def forward_multimodal(self, x_vis, x_therm, extrinsics=None, *args, **kwargs):
        """
        执行模型的前向传播。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            extrinsics (torch.Tensor): 外参矩阵。
            *args: 额外的位置参数。
            **kwargs: 额外的关键字参数。

        Returns:
            torch.Tensor: 模型的输出。
        """
        if isinstance(x_vis, dict) and isinstance(x_therm, dict):
            return self.loss(x_vis, x_therm, extrinsics,*args, **kwargs)
        return self.predict(x_vis, x_therm, extrinsics, *args, **kwargs)

    def loss(self, x_vis, x_therm, extrinsics=None, *args, **kwargs):
        """
        计算损失。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            *args: 额外的位置参数。
            **kwargs: 额外的关键字参数。

        Returns:
            torch.Tensor: 计算得到的损失。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        # print(x_vis.keys(), ", ", x_therm.keys())
        preds = self.forward(x_vis["img"], x_therm["img2"], extrinsics)  # 前向传播
        return self.criterion(preds, x_vis)

    def predict(self, x_vis, x_therm, extrinsics=None, profile=False, visualize=False, augment=False, embed=None):
        """
        执行模型的预测。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            extrinsics (torch.Tensor): 外参矩阵。
            profile (bool): 是否打印每层的计算时间。
            visualize (bool): 是否保存特征图用于可视化。
            augment (bool): 是否进行数据增强预测。
            embed (list, optional): 要返回的特征向量列表。

        Returns:
            torch.Tensor: 模型的预测输出。
        """
        if augment:
            return self._predict_augment(x_vis, x_therm, extrinsics)
        return self._predict_once(x_vis, x_therm, extrinsics, profile, visualize, embed)

    def _predict_once_v1(self, x_vis, x_therm, profile=False, visualize=False, embed=None):
        """
        执行一次前向传播进行预测。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            profile (bool): 是否打印每层的计算时间。
            visualize (bool): 是否保存特征图用于可视化。
            embed (list, optional): 要返回的特征向量列表。

        Returns:
            torch.Tensor: 模型的预测输出。
        """
        y, dt, embeddings = [], [], []  # 输出
        # print("self.model: ", self.model)
        for m in self.model:
            print("m.i: ", m.i, "self.yaml['backbone']: ", len(self.yaml['backbone']))
            print("x_vis.shape: ", x_vis.shape, "x_therm.shape: ", x_therm.shape)
            if m.i < len(self.yaml['backbone']):
                x_vis = m(x_vis)
                x_therm = x_therm
                y.append(x_vis if m.i in self.save else None)
            else:
                if m.f != -1:  # 如果不是来自上一层
                    x_therm = y[m.f] if isinstance(m.f, int) else [x_therm if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x_therm, dt)
                # print("model: ", m, "x_therm.shape: ", x_therm.shape)
                x_therm = m(x_therm)  # 运行
                x_vis = x_vis
                y.append(x_therm if m.i in self.save else None)  # 保存输出
                
            if visualize:
                feature_visualization(x_therm, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x_therm, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x_therm

    def _predict_once(self, x_vis, x_therm, extrinsics=None, profile=False, visualize=False, embed=None):
        """
        执行一次前向传播进行预测。

        Args:
            x_vis (torch.Tensor): 可见光输入图像张量。
            x_therm (torch.Tensor): 红外输入图像张量。
            extrinsics (torch.Tensor): 外参矩阵。
            profile (bool): 是否打印每层的计算时间。
            visualize (bool): 是否保存特征图用于可视化。
            embed (list, optional): 要返回的特征向量列表。

        Returns:
            torch.Tensor: 模型的预测输出。
        """
        # 同时获取多批次图像尺寸大小用来传入EnhancedFMDEA
        B = x_vis.size(0)
        original_sizes = torch.zeros(B, 2, 2, device=x_vis.device, dtype=x_vis.dtype)
        
        # 存储每个样本的RGB和IR图像尺寸
        for b in range(B):
            original_sizes[b, 0] = torch.tensor([x_vis[b].shape[-2], x_vis[b].shape[-1]])  # RGB: (H, W)
            original_sizes[b, 1] = torch.tensor([x_therm[b].shape[-2], x_therm[b].shape[-1]])  # IR: (H, W)

        y_main = [x_vis]   # 可见光分支各层输出，y_main[0]为初始输入
        y_aux = [x_therm]  # 红外分支各层输出，y_aux[0]为初始输入
        y_head = []        # 头部各层输出
        dt = []            # 性能分析时间记录
        embeddings = []    # 嵌入特征

        backbone_len = len(self.yaml['backbone'])
        backbone2_len = len(self.yaml['backbone2'])

        for m in self.model:
            # 处理Backbone（可见光分支）
            if m.i < backbone_len:
                f = m.f
                # 获取输入特征
                if isinstance(f, int):
                    x = y_main[f] if f != -1 else y_main[-1]
                elif isinstance(f, list):
                    x = [y_main[i] for i in f]
                else:
                    x = y_main[-1]
                # 执行模块
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
                y_main.append(x)  # 保存输出到可见光分支

            # 处理Backbone2（红外分支）
            elif m.i < backbone_len + backbone2_len:
                f = m.f
                # 转换为backbone2内部的索引
                if isinstance(f, int):
                    f_internal = f if f != -1 else len(y_aux) - 1
                    x = y_aux[f_internal]
                elif isinstance(f, list):
                    x = [y_aux[i] for i in f]
                else:
                    x = y_aux[-1]
                # 执行模块
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
                y_aux.append(x)  # 保存输出到红外分支

            # 处理Head（融合及检测头）
            else:
                f = m.f
                inputs = []
                # 遍历输入来源索引
                for src_idx in (f if isinstance(f, list) else [f]):
                    if src_idx == -1:
                        inputs.append(y_head[src_idx])
                    elif src_idx < backbone_len:  # 来自可见光分支
                        inputs.append(y_main[src_idx + 1])  # y_main[0]为输入，层0输出在y_main[1]
                    elif src_idx < backbone_len + backbone2_len:  # 来自红外分支
                        layer_idx = src_idx - backbone_len
                        inputs.append(y_aux[layer_idx + 1])  # y_aux[0]为输入，层0输出在y_aux[1]
                    else:  # 来自头部之前的层
                        head_idx = src_idx - (backbone_len + backbone2_len)
                        inputs.append(y_head[head_idx])
                # 处理多输入情况（如Concat或DEA）
                if isinstance(m, EnhancedFMDEA):
                    inputs.append(extrinsics)
                    inputs.append(original_sizes)
                x = inputs if len(inputs) > 1 else inputs[0] if inputs else None
                # 执行模块
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
                y_head.append(x)  # 保存头部输出

                # 可视化与特征嵌入
                if visualize:
                    feature_visualization(x, m.__class__.__name__, m.i, save_dir=visualize)
                if embed and m.i in embed:
                    embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                    if m.i == max(embed):
                        return torch.unbind(torch.cat(embeddings, 1), 0)

        # 最终输出为头部的最后一层
        return y_head[-1] if y_head else None

    def _predict_augment_origin(self, x_vis, x_therm, extrinsics=None):
        """对输入的可见光和红外图像进行数据增强预测，并返回增强后的推理结果。"""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "FusionDetectionModel":
            LOGGER.warning("WARNING ⚠️ 模型不支持 'augment=True'，恢复为单尺度预测。")
            return self._predict_once(x_vis, x_therm)
        
        img_size = x_vis.shape[-2:]  # 高度, 宽度
        scales = [1, 0.83, 0.67]  # 缩放比例
        flips = [None, 3, None]  # 翻转方式（None, 水平翻转, None）
        augmented_preds = []

        for scale, flip in zip(scales, flips):
            # 对可见光图像进行翻转和缩放
            if flip:
                x_vis_aug = scale_img(x_vis.flip(flip), scale, gs=int(self.stride.max()))
                x_therm_aug = scale_img(x_therm.flip(flip), scale, gs=int(self.stride.max()))
            else:
                x_vis_aug = scale_img(x_vis, scale, gs=int(self.stride.max()))
                x_therm_aug = scale_img(x_therm, scale, gs=int(self.stride.max()))
            
            # 前向推理
            preds = super().predict(x_vis_aug, x_therm_aug)[0]
            # 反转换预测结果
            preds = self._descale_pred(preds, flip, scale, img_size)
            augmented_preds.append(preds)
        
        # 裁剪增强后的预测结果
        augmented_preds = self._clip_augmented(augmented_preds)
        return torch.cat(augmented_preds, -1), None  # 增强后的推理结果和训练输出

    def _predict_augment(self, x_vis, x_therm, extrinsics=None):
        """
        对输入的可见光和红外图像进行数据增强预测，并同步更新对应的单应性矩阵。
        
        Args:
            x_vis (torch.Tensor): 可见光图像张量
            x_therm (torch.Tensor): 红外图像张量
            extrinsics (torch.Tensor): 原始单应性矩阵, shape为(B, 3, 3)
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "MultiModalDetectionModel":
            LOGGER.warning("WARNING ⚠️ 模型不支持 'augment=True'，恢复为单尺度预测。")
            return self._predict_once(x_vis, x_therm)
        
        img_size = x_vis.shape[-2:]  # 高度, 宽度
        scales = [1, 0.83, 0.67]  # 缩放比例
        flips = [None, 3, None]  # 翻转方式（None, 水平翻转, None）
        augmented_preds = []
        augmented_matrices = []

        if extrinsics is None:
            extrinsics = torch.eye(3, device=x_vis.device).expand(x_vis.shape[0], -1, -1)

        for scale, flip in zip(scales, flips):
            # 1. 记录原始关键点
            original_points = self.get_keypoints(x_vis)
            original_mapped_points = self.transform_points(original_points, extrinsics)

            # 2. 进行数据增强
            if flip:
                x_vis_aug = scale_img(x_vis.flip(flip), scale, gs=int(self.stride.max()))
                x_therm_aug = scale_img(x_therm.flip(flip), scale, gs=int(self.stride.max()))
                # 更新单应性矩阵以反映水平翻转
                flip_matrix = torch.eye(3, device=extrinsics.device)
                if flip == 3:  # 水平翻转
                    flip_matrix[0, 0] = -1
                    flip_matrix[0, 2] = img_size[1]
            else:
                x_vis_aug = scale_img(x_vis, scale, gs=int(self.stride.max()))
                x_therm_aug = scale_img(x_therm, scale, gs=int(self.stride.max()))
                matrix_aug = extrinsics.clone()

            # 3. 更新缩放变换
            scale_matrix = torch.eye(3, device=matrix_aug.device)
            scale_matrix[0, 0] = scale_matrix[1, 1] = scale
            matrix_aug = matrix_aug @ scale_matrix

            # 4. 检查变换一致性
            if not self.check_transformation_consistency(
                x_vis_aug, x_therm_aug, matrix_aug,
                original_points, original_mapped_points,
                scale, flip, img_size
            ):
                LOGGER.warning(f"变换一致性检查失败: scale={scale}, flip={flip}")
                continue

            # 5. 前向推理
            preds = super().predict(x_vis_aug, x_therm_aug)[0]
            
            # 6. 反变换
            preds = self._descale_pred(preds, flip, scale, img_size)
            matrix_aug = self._descale_matrix(matrix_aug, flip, scale, img_size)
            
            # 7. 再次检查反变换后的一致性
            if not self.check_transformation_consistency(
                x_vis, x_therm, matrix_aug,
                original_points, original_mapped_points,
                1.0, None, img_size
            ):
                LOGGER.warning("反变换后的一致性检查失败")
                continue
            
            augmented_preds.append(preds)
            augmented_matrices.append(matrix_aug)
        
        # 8. 最终检查所有变换结果
        if len(augmented_matrices) > 0:
            final_consistency = all(
                torch.allclose(
                    self.transform_points(original_points, aug_matrix),
                    original_mapped_points,
                    rtol=1e-3
                )
                for aug_matrix in augmented_matrices
            )
            if not final_consistency:
                LOGGER.warning("最终变换一致性检查失败")
        
        # 裁剪增强后的预测结果
        augmented_preds = self._clip_augmented(augmented_preds)
        
        return torch.cat(augmented_preds, -1), torch.stack(augmented_matrices, 1)

    def check_transformation_consistency(self, x_vis, x_therm, H, 
                                      original_points, original_mapped_points,
                                      scale, flip, img_size, rtol=1e-3):
        """
        检查变换前后的映射关系是否保持一致
        
        Args:
            x_vis: 变换后的可见光图像
            x_therm: 变换后的红外图像
            H: 变换后的单应性矩阵
            original_points: 原始图像中的关键点
            original_mapped_points: 原始关键点经过原始单应性矩阵映射后的点
            scale: 缩放比例
            flip: 翻转类型
            img_size: 原始图像尺寸
            rtol: 相对误差容限
        """
        # 1. 获取变换后图像中的关键点
        transformed_points = self.transform_points(original_points, H)
        
        # 2. 检查关键点映射关系
        if not torch.allclose(transformed_points, original_mapped_points, rtol=rtol):
            return False
        
        # 3. 检查图像尺寸一致性
        if x_vis.shape[-2:] != x_therm.shape[-2:]:
            return False
        
        # 4. 检查变换矩阵的有效性
        if not self.validate_homography(H):
            return False
        
        return True

    def validate_homography(self, H, eps=1e-6):
        """
        验证单应性矩阵的有效性
        """
        # 检查数值是否有效
        if not torch.isfinite(H).all():
            return False
            
        # 检查行列式是否接近于0
        if torch.abs(torch.det(H)) < eps:
            return False
            
        # 检查最后一行是否接近 [0,0,1]
        if not torch.allclose(H[2], torch.tensor([0., 0., 1.], device=H.device), rtol=1e-3):
            return False
            
        return True

    def get_keypoints(self, image, n_points=4):
        """
        获取图像中的关键点（例如角点）
        """
        h, w = image.shape[-2:]
        points = torch.tensor([
            [0, 0, 1],
            [w, 0, 1],
            [0, h, 1],
            [w, h, 1]
        ], device=image.device, dtype=torch.float32)
        return points

    def transform_points(self, points, H):
        """
        使用单应性矩阵变换点坐标
        """
        # points: (N, 3)
        # H: (3, 3)
        transformed = (H @ points.T).T
        # 归一化齐次坐标
        transformed = transformed / transformed[:, 2:3]
        return transformed

    def init_criterion(self):
        """初始化 FusionDetectionModel 的损失函数."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {Classify,Conv,ConvTranspose,GhostConv,Bottleneck,GhostBottleneck,SPP,SPPF,C2fPSA,C2PSA,DWConv,
                 Focus,BottleneckCSP,C1,C2,C2f,C3k2,RepNCSPELAN4,ELAN1,ADown,AConv,SPPELAN,C2fAttn,C3,C3TR,
                 C3Ghost,nn.ConvTranspose2d,DWConvTranspose2d,C3x,RepC3,PSA,SCDown,C2fCIB,}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP,C1,C2,C2f,C3k2,C2fAttn,C3,C3TR,C3Ghost,}:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m in {CBLinear, TorchVision, Index}:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def parse_model_fusion(d, ch, ch2, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Initial parameters
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, max_channels = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "max_channels"))
    if scales:
        scale = d.get("scale", list(scales.keys())[0])
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    # Separate layers for backbone, backbone2, and head
    backbone_layers = d["backbone"]
    backbone2_layers = d["backbone2"]
    head_layers = d["head"]
    backbone_len = len(backbone_layers)
    backbone2_len = len(backbone2_layers)

    # Initialize channel lists for each branch
    ch_main = [ch]   # Visible branch channels
    ch_aux = [ch2]   # Infrared branch channels
    ch_head = []
    layers = []
    save = []

    # Process backbone (visible branch)
    for i, (f, n, m, args) in enumerate(backbone_layers):
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        # Get input and output channels
        if m in [Conv, C3k2, SPPF, C2PSA]:
            c1 = ch_main[f] if isinstance(f, int) else [ch_main[x] for x in f]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m is C3k2:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True

        # Build module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t
        layers.append(m_)
        ch_main.append(c2)
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print

    # Process backbone2 (infrared branch)
    for i, (f, n, m, args) in enumerate(backbone2_layers):
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        # Get input and output channels
        if m in [Conv, C3k2, SPPF, C2PSA]:
            c1 = ch_aux[f] if isinstance(f, int) else [ch_aux[x] for x in f]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m is C3k2:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True

        # Build module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + backbone_len, f, t  # Offset index for backbone2
        layers.append(m_)
        ch_aux.append(c2)
        if verbose:
            LOGGER.info(f"{i + backbone_len:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print

    # Process head (fusion and detection)
    for i, (f, n, m, args) in enumerate(head_layers):
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if m in {Classify,Conv,ConvTranspose,GhostConv,Bottleneck,GhostBottleneck,SPP,SPPF,C2fPSA,C2PSA,DWConv,
                 Focus,BottleneckCSP,C1,C2,C2f,C3k2,RepNCSPELAN4,ELAN1,ADown,AConv,SPPELAN,C2fAttn,C3,C3TR,
                 C3Ghost,nn.ConvTranspose2d,DWConvTranspose2d,C3x,RepC3,PSA,SCDown,C2fCIB,}:
            c1, c2 = ch_head[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP,C1,C2,C2f,C3k2,C2fAttn,C3,C3TR,C3Ghost,}:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is DEA:
            c1 = []
            for x in (f if isinstance(f, list) else [f]):
                if x < backbone_len:  # From visible branch
                    c1.append(ch_main[x + 1])
                else:  # From infrared branch (adjust index)
                    c1.append(ch_aux[x - backbone_len + 1])
                
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]  # Pass both features to DEA
        elif m in {FMDEA, EnhancedFMDEA}:
            c1 = []
            for x in (f if isinstance(f, list) else [f]):
                if x < backbone_len:  # From visible branch
                    c1.append(ch_main[x + 1])
                else:  # From infrared branch (adjust index)
                    c1.append(ch_aux[x - backbone_len + 1])
            # mapping_matrix = d["mapping_matrix"]
            # print(f'mapping_matrix:{mapping_matrix}')
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]  # Pass both features to DEA
        elif m is Concat:
            c1 = []
            for x in (f if isinstance(f, list) else [f]):
                if x >= len(ch_main) + len(ch_aux) - 2:
                    c1.append(ch_head[x - len(ch_main) - len(ch_aux) + 2])
            c2 = sum(c1)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch_head[x - len(ch_main) - len(ch_aux) + 2] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        else:
            # print(f'm:{m}')
            # c2 = args[0]
            c2 = ch_head[f]

        # Build module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + backbone_len + backbone2_len, f, t
        layers.append(m_)
        ch_head.append(c2)
        if verbose:
            LOGGER.info(f"{i + backbone_len + backbone2_len:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print

    return nn.Sequential(*layers), sorted(save)



def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect

def check_transformation_consistency(x_vis, x_therm, H, transformed_x_vis, transformed_x_therm, transformed_H):
    """检查变换前后的一致性"""
    # 选择关键点进行验证
    points_vis = torch.tensor([[0, 0, 1], [x_vis.shape[2], x_vis.shape[3], 1]], device=x_vis.device)
    points_therm = (H @ points_vis.T).T
    
    # 变换后的点
    transformed_points_vis = torch.tensor([[0, 0, 1], [transformed_x_vis.shape[2], transformed_x_vis.shape[3], 1]], device=x_vis.device)
    transformed_points_therm = (transformed_H @ transformed_points_vis.T).T
    
    # 检查映射关系是否保持
    return torch.allclose(points_therm, transformed_points_therm, rtol=1e-3)
