# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from typing import Dict, List, Optional, Tuple, Union, Callable

class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """Check image caching requirements vs available disk space."""
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = cv2.imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError


class BaseMultiModalDataset(Dataset):
    """
    多模态数据集基类，用于加载和处理图像数据。

    Args:
        img_path (str): 第一模态图像文件夹路径。
        img_path2 (str): 第二模态图像文件夹路径。
        imgsz (int, optional): 图像尺寸。默认值为 640。
        cache (Union[bool, str], optional): 是否缓存图像到 RAM 或磁盘。可以为 True, False, "ram", "disk"。默认值为 False。
        cache2 (Union[bool, str], optional): 第二模态的缓存选项。默认值为 False。
        augment (bool, optional): 是否应用数据增强。默认值为 True。
        hyp (dict, optional): 数据增强的超参数。默认值为 DEFAULT_CFG。
        prefix (str, optional): 日志前缀。默认值为空字符串。
        prefix2 (str, optional): 第二模态的日志前缀。默认值为空字符串。
        rect (bool, optional): 是否使用矩形训练。默认值为 False。
        batch_size (int, optional): 批次大小。默认值为 16。
        stride (int, optional): 步幅。默认值为 32。
        pad (float, optional): 填充比例。默认值为 0.5。
        single_cls (bool, optional): 是否进行单类别训练。默认值为 False。
        classes (Optional[List[int]]): 包含的类别列表。默认值为 None。
        fraction (float, optional): 使用的数据集比例。默认值为 1.0（使用全部数据）。

    Attributes:
        im_files (List[str]): 第一模态的图像文件路径列表。
        im_files2 (List[str]): 第二模态的图像文件路径列表。
        labels (List[Dict]): 标签数据字典列表。
        ni (int): 数据集中图像数量。
        ims (List[Optional[np.ndarray]]): 第一模态已加载的图像。
        ims2 (List[Optional[np.ndarray]]): 第二模态已加载的图像。
        im_hw0 (List[Optional[Tuple[int, int]]]): 第一模态原始图像尺寸。
        im_hw02 (List[Optional[Tuple[int, int]]]): 第二模态原始图像尺寸。
        im_hw (List[Optional[Tuple[int, int]]]: 第一模态调整后的图像尺寸。
        im_hw2 (List[Optional[Tuple[int, int]]]: 第二模态调整后的图像尺寸。
        npy_files (List[Path]): 第一模态的 numpy 文件路径列表。
        npy_files2 (List[Path]): 第二模态的 numpy 文件路径列表。
        cache (Optional[str]): 第一模态的缓存类型。
        cache2 (Optional[str]): 第二模态的缓存类型。
        transforms (Callable): 图像变换函数。
        buffer (List): 缓冲区，用于拼接图像。
        max_buffer_length (int): 缓冲区最大长度。
    """

    def __init__(
        self,
        img_path: str,
        img_path2: str,
        imgsz: int = 640,
        cache: Union[bool, str] = False,
        cache2: Union[bool, str] = False,
        augment: bool = True,
        hyp: Dict = DEFAULT_CFG,
        prefix: str = "",
        prefix2: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: Optional[List[int]] = None,
        fraction: float = 1.0,
    ):
        """初始化 BaseMultiModalDataset 类。"""
        super().__init__()
        self.img_path = img_path
        self.img_path2 = img_path2
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.prefix2 = prefix2
        self.fraction = fraction

        # 获取图像文件
        self.im_files = self.get_img_files(self.img_path)
        self.im_files2 = self.get_img_files(self.img_path2)

        # 获取标签
        self.labels = self.get_labels()

        # 更新标签，考虑包含的类别
        self.update_labels(include_class=classes)

        self.ni = len(self.labels)  # 图像数量
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad

        if self.rect:
            assert self.batch_size is not None, "batch_size 必须在 rect 模式下指定。"
            self.set_rectangle()

        # 缓冲区，用于拼接图像
        self.buffer: List = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 缓存图像
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        if cache2 == 'ram' and not self.check_cache_ram2():
            cache2 = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.ims2, self.im_hw02, self.im_hw2 = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.npy_files2 = [Path(f).with_suffix(".npy") for f in self.im_files2]
        if cache:
            self.cache_images(cache)
        if cache2:
            self.cache_images2(cache2)

        # 图像变换
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                                interpolation=interp)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def load_image2(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims2[i], self.im_files2[i], self.npy_files2[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                                interpolation=interp)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims2[i], self.im_hw02[i], self.im_hw2[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims2[j], self.im_hw02[j], self.im_hw2[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims2[i], self.im_hw02[i], self.im_hw2[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images2(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk2 if cache == 'disk' else self.load_image2
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files2[i].stat().st_size
                else:  # 'ram'
                    self.ims2[i], self.im_hw02[i], self.im_hw2[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims2[i].nbytes
                pbar.desc = f'{self.prefix2}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def cache_images_to_disk2(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files2[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files2[i]))

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def check_cache_ram2(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files2))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix2}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.im_files2 = [self.im_files2[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['img2'], label['ori_shape2'], label['resized_shape2'] = self.load_image2(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label['ratio_pad2'] = (label['resized_shape2'][0] / label['ori_shape2'][0],
                               label['resized_shape2'][1] / label['ori_shape2'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError


# class BaseMultiModalDataset(Dataset):
#     """
#     Base dataset class for loading and processing image data.

#     Args:
#         img_path (str): Path to the folder containing images.
#         imgsz (int, optional): Image size. Defaults to 640.
#         cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
#         augment (bool, optional): If True, data augmentation is applied. Defaults to True.
#         hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
#         prefix (str, optional): Prefix to print in log messages. Defaults to ''.
#         rect (bool, optional): If True, rectangular training is used. Defaults to False.
#         batch_size (int, optional): Size of batches. Defaults to None.
#         stride (int, optional): Stride. Defaults to 32.
#         pad (float, optional): Padding. Defaults to 0.0.
#         single_cls (bool, optional): If True, single class training is used. Defaults to False.
#         classes (list): List of included classes. Default is None.
#         fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

#     Attributes:
#         im_files (list): List of image file paths.
#         labels (list): List of label data dictionaries.
#         ni (int): Number of images in the dataset.
#         ims (list): List of loaded images.
#         npy_files (list): List of numpy file paths.
#         transforms (callable): Image transformation function.
#     """

#     def __init__(
#         self,
#         img_path,
#         img_path2,      #  多模态
#         imgsz=640,
#         cache=False,
#         cache2=False,   #  多模态
#         augment=True,
#         hyp=DEFAULT_CFG,
#         prefix="",
#         prefix2="",     #  多模态
#         rect=False,
#         batch_size=16,
#         stride=32,
#         pad=0.5,
#         single_cls=False,
#         classes=None,
#         fraction=1.0,
#     ):
#         """Initialize BaseDataset with given configuration and options."""
#         super().__init__()
#         self.img_path = img_path
#         self.img_path2 = img_path2
#         self.imgsz = imgsz
#         self.augment = augment
#         self.single_cls = single_cls
#         self.prefix = prefix
#         self.prefix2 = prefix2
#         self.fraction = fraction
#         self.im_files = self.get_img_files(self.img_path)
#         self.im_files2 = self.get_img_files(self.img_path2)
#         self.labels = self.get_labels()
#         # self.labels2 = self.get_labels2()
#         self.update_labels(include_class=classes)  # single_cls and include_class
#         self.ni = len(self.labels)  # number of images
#         self.rect = rect
#         self.batch_size = batch_size
#         self.stride = stride
#         self.pad = pad
#         if self.rect:
#             assert self.batch_size is not None
#             self.set_rectangle()

#         # Buffer thread for mosaic images
#         self.buffer = []  # buffer size = batch size
#         self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

#         # Cache images (options are cache = True, False, None, "ram", "disk")
#         self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
#         self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
#         self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
#         if self.cache == "ram" and self.check_cache_ram():
#             if hyp.deterministic:
#                 LOGGER.warning(
#                     "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
#                     "Consider cache='disk' as a deterministic alternative if your disk space allows."
#                 )
#             self.cache_images()
#         elif self.cache == "disk" and self.check_cache_disk():
#             self.cache_images()
        
#         # 多模态
#         self.ims2, self.im_hw02, self.im_hw2 = [None] * self.ni, [None] * self.ni, [None] * self.ni
#         self.npy_files2 = [Path(f).with_suffix(".npy") for f in self.im_files2]
#         self.cache2 = cache2.lower() if isinstance(cache2, str) else "ram" if cache2 is True else None
#         if self.cache2 == "ram" and self.check_cache_ram():
#             if hyp.deterministic:
#                 LOGGER.warning(
#                     "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
#                     "Consider cache='disk' as a deterministic alternative if your disk space allows."
#                 )
#             self.cache_images()
#         elif self.cache2 == "disk" and self.check_cache_disk():
#             self.cache_images()

#         # Transforms
#         self.transforms = self.build_transforms(hyp=hyp)

#     def get_img_files(self, img_path):
#         """Read image files."""
#         try:
#             f = []  # image files
#             for p in img_path if isinstance(img_path, list) else [img_path]:
#                 p = Path(p)  # os-agnostic
#                 if p.is_dir():  # dir
#                     f += glob.glob(str(p / "**" / "*.*"), recursive=True)
#                     # F = list(p.rglob('*.*'))  # pathlib
#                 elif p.is_file():  # file
#                     with open(p) as t:
#                         t = t.read().strip().splitlines()
#                         parent = str(p.parent) + os.sep
#                         f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
#                         # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
#                 else:
#                     raise FileNotFoundError(f"{self.prefix}{p} does not exist")
#             im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
#             # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
#             assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
#         except Exception as e:
#             raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
#         if self.fraction < 1:
#             im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
#         return im_files

#     def update_labels(self, include_class: Optional[list]):
#         """Update labels to include only these classes (optional)."""
#         include_class_array = np.array(include_class).reshape(1, -1)
#         for i in range(len(self.labels)):
#             if include_class is not None:
#                 cls = self.labels[i]["cls"]
#                 bboxes = self.labels[i]["bboxes"]
#                 segments = self.labels[i]["segments"]
#                 keypoints = self.labels[i]["keypoints"]
#                 j = (cls == include_class_array).any(1)
#                 self.labels[i]["cls"] = cls[j]
#                 self.labels[i]["bboxes"] = bboxes[j]
#                 if segments:
#                     self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
#                 if keypoints is not None:
#                     self.labels[i]["keypoints"] = keypoints[j]
#             if self.single_cls:
#                 self.labels[i]["cls"][:, 0] = 0

#     def load_image(self, i, rect_mode=True):
#         """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
#         im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
#         if im is None:  # not cached in RAM
#             if fn.exists():  # load npy
#                 try:
#                     im = np.load(fn)
#                 except Exception as e:
#                     LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
#                     Path(fn).unlink(missing_ok=True)
#                     im = cv2.imread(f)  # BGR
#             else:  # read image
#                 im = cv2.imread(f)  # BGR
#             if im is None:
#                 raise FileNotFoundError(f"Image Not Found {f}")

#             h0, w0 = im.shape[:2]  # orig hw
#             if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
#                 r = self.imgsz / max(h0, w0)  # ratio
#                 if r != 1:  # if sizes are not equal
#                     w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
#                     im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
#             elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
#                 im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

#             # Add to buffer if training with augmentations
#             if self.augment:
#                 self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#                 self.buffer.append(i)
#                 if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
#                     j = self.buffer.pop(0)
#                     if self.cache != "ram":
#                         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

#             return im, (h0, w0), im.shape[:2]

#         return self.ims[i], self.im_hw0[i], self.im_hw[i]

#     def load_image2(self, i, rect_mode=True):
#         """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
#         im, f, fn = self.ims2[i], self.im_files2[i], self.npy_files2[i]
#         if im is None:  # not cached in RAM
#             if fn.exists():  # load npy
#                 try:
#                     im = np.load(fn)
#                 except Exception as e:
#                     LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
#                     Path(fn).unlink(missing_ok=True)
#                     im = cv2.imread(f)  # BGR
#             else:  # read image
#                 im = cv2.imread(f)  # BGR
#             if im is None:
#                 raise FileNotFoundError(f"Image Not Found {f}")

#             h0, w0 = im.shape[:2]  # orig hw
#             if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
#                 r = self.imgsz / max(h0, w0)  # ratio
#                 if r != 1:  # if sizes are not equal
#                     w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
#                     im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
#             elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
#                 im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

#             # Add to buffer if training with augmentations
#             if self.augment:
#                 self.ims2[i], self.im_hw02[i], self.im_hw2[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#                 self.buffer.append(i)
#                 if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
#                     j = self.buffer.pop(0)
#                     if self.cache2 != "ram":
#                         self.ims2[j], self.im_hw02[j], self.im_hw2[j] = None, None, None

#             return im, (h0, w0), im.shape[:2]

#         return self.ims2[i], self.im_hw02[i], self.im_hw2[i]

#     def cache_images(self):
#         """Cache images to memory or disk."""
#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
#         with ThreadPool(NUM_THREADS) as pool:
#             results = pool.imap(fcn, range(self.ni))
#             pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
#             for i, x in pbar:
#                 if self.cache == "disk":
#                     b += self.npy_files[i].stat().st_size
#                 else:  # 'ram'
#                     self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
#                     b += self.ims[i].nbytes
#                 pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
#             pbar.close()
    
#     def cache_images2(self):
#         """Cache images to memory or disk."""
#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         fcn, storage = (self.cache_images_to_disk2, "Disk") if self.cache2 == "disk" else (self.load_image2, "RAM")
#         with ThreadPool(NUM_THREADS) as pool:
#             results = pool.imap(fcn, range(self.ni))
#             pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
#             for i, x in pbar:
#                 if self.cache2 == "disk":
#                     b += self.npy_files2[i].stat().st_size
#                 else:  # 'ram'
#                     self.ims2[i], self.im_hw02[i], self.im_hw2[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
#                     b += self.ims2[i].nbytes
#                 pbar.desc = f"{self.prefix2}Caching images ({b / gb:.1f}GB {storage})"
#             pbar.close()

#     def cache_images_to_disk(self, i):
#         """Saves an image as an *.npy file for faster loading."""
#         f = self.npy_files[i]
#         if not f.exists():
#             np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

#     def cache_images_to_disk2(self, i):
#         """Saves an image as an *.npy file for faster loading."""
#         f = self.npy_files2[i]
#         if not f.exists():
#             np.save(f.as_posix(), cv2.imread(self.im_files2[i]), allow_pickle=False)

#     def check_cache_disk(self, safety_margin=0.5):
#         """Check image caching requirements vs available disk space."""
#         import shutil

#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         n = min(self.ni, 30)  # extrapolate from 30 random images
#         for _ in range(n):
#             im_file = random.choice(self.im_files)
#             im = cv2.imread(im_file)
#             if im is None:
#                 continue
#             b += im.nbytes
#             if not os.access(Path(im_file).parent, os.W_OK):
#                 self.cache = None
#                 LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
#                 return False
#         disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
#         total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
#         if disk_required > free:
#             self.cache = None
#             LOGGER.info(
#                 f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
#                 f"with {int(safety_margin * 100)}% safety margin but only "
#                 f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
#             )
#             return False
#         return True

#     def check_cache_disk2(self, safety_margin=0.5):
#         """Check image caching requirements vs available disk space."""
#         import shutil

#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         n = min(self.ni, 30)  # extrapolate from 30 random images
#         for _ in range(n):
#             im_file = random.choice(self.im_files)
#             im = cv2.imread(im_file)
#             if im is None:
#                 continue
#             b += im.nbytes
#             if not os.access(Path(im_file).parent, os.W_OK):
#                 self.cache2 = None
#                 LOGGER.info(f"{self.prefix2}Skipping caching images to disk, directory not writeable ⚠️")
#                 return False
#         disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
#         total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
#         if disk_required > free:
#             self.cache2 = None
#             LOGGER.info(
#                 f"{self.prefix2}{disk_required / gb:.1f}GB disk space required, "
#                 f"with {int(safety_margin * 100)}% safety margin but only "
#                 f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
#             )
#             return False
#         return True

#     def check_cache_ram(self, safety_margin=0.5):
#         """Check image caching requirements vs available memory."""
#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         n = min(self.ni, 30)  # extrapolate from 30 random images
#         for _ in range(n):
#             im = cv2.imread(random.choice(self.im_files))  # sample image
#             if im is None:
#                 continue
#             ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
#             b += im.nbytes * ratio**2
#         mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
#         mem = psutil.virtual_memory()
#         if mem_required > mem.available:
#             self.cache = None
#             LOGGER.info(
#                 f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
#                 f"with {int(safety_margin * 100)}% safety margin but only "
#                 f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
#             )
#             return False
#         return True

#     def check_cache_ram2(self, safety_margin=0.5):
#         """Check image caching requirements vs available memory."""
#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         n = min(self.ni, 30)  # extrapolate from 30 random images
#         for _ in range(n):
#             im = cv2.imread(random.choice(self.im_files2))  # sample image
#             if im is None:
#                 continue
#             ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
#             b += im.nbytes * ratio**2
#         mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
#         mem = psutil.virtual_memory()
#         if mem_required > mem.available:
#             self.cache2 = None
#             LOGGER.info(
#                 f"{self.prefix2}{mem_required / gb:.1f}GB RAM required to cache images "
#                 f"with {int(safety_margin * 100)}% safety margin but only "
#                 f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
#             )
#             return False
#         return True

#     def set_rectangle(self):
#         """Sets the shape of bounding boxes for YOLO detections as rectangles."""
#         bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
#         nb = bi[-1] + 1  # number of batches

#         s = np.array([x.pop("shape") for x in self.labels])  # hw
#         ar = s[:, 0] / s[:, 1]  # aspect ratio
#         irect = ar.argsort()
#         self.im_files = [self.im_files[i] for i in irect]
#         self.labels = [self.labels[i] for i in irect]
#         ar = ar[irect]

#         # Set training image shapes
#         shapes = [[1, 1]] * nb
#         for i in range(nb):
#             ari = ar[bi == i]
#             mini, maxi = ari.min(), ari.max()
#             if maxi < 1:
#                 shapes[i] = [maxi, 1]
#             elif mini > 1:
#                 shapes[i] = [1, 1 / mini]

#         self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
#         self.batch = bi  # batch index of image

#     def __getitem__(self, index):
#         """Returns transformed label information for given index."""
#         return self.transforms(self.get_image_and_label(index))

#     def get_image_and_label(self, index):
#         """Get and return label information from the dataset."""
#         label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
#         label.pop("shape", None)  # shape is for rect, remove it
#         label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
#         label['img2'], label['ori_shape2'], label['resized_shape2'] = self.load_image2(index)
#         label["ratio_pad"] = (
#             label["resized_shape"][0] / label["ori_shape"][0],
#             label["resized_shape"][1] / label["ori_shape"][1],
#         )  # for evaluation
#         label["ratio_pad2"] = (
#             label["resized_shape2"][0] / label["ori_shape2"][0],
#             label["resized_shape2"][1] / label["ori_shape2"][1],
#         )  # for evaluation
#         if self.rect:
#             label["rect_shape"] = self.batch_shapes[self.batch[index]]
#         return self.update_labels_info(label)

#     def __len__(self):
#         """Returns the length of the labels list for the dataset."""
#         return len(self.labels)

#     def update_labels_info(self, label):
#         """Custom your label format here."""
#         return label

#     def build_transforms(self, hyp=None):
#         """
#         Users can customize augmentations here.

#         Example:
#             ```python
#             if self.augment:
#                 # Training transforms
#                 return Compose([])
#             else:
#                 # Val transforms
#                 return Compose([])
#             ```
#         """
#         raise NotImplementedError

#     def get_labels(self):
#         """
#         Users can customize their own format here.

#         Note:
#             Ensure output is a dictionary with the following keys:
#             ```python
#             dict(
#                 im_file=im_file,
#                 shape=shape,  # format: (height, width)
#                 cls=cls,
#                 bboxes=bboxes,  # xywh
#                 segments=segments,  # xy
#                 keypoints=keypoints,  # xy
#                 normalized=True,  # or False
#                 bbox_format="xyxy",  # or xywh, ltwh
#             )
#             ```
#         """
#         raise NotImplementedError
