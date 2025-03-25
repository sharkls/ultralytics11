# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class MultiModalDetectionPredictor(BasePredictor):    
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.multimodal import MultiModalDetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = MultiModalDetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, img_type=0):
        """
        Post-processes predictions and returns a list of Results objects.
        
        Args:
            preds: 模型预测结果
            img: 预处理后的图像
            orig_imgs: 原始图像
            img_type: 0表示可见光，1表示红外
        """
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )       

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        # 获取对应的原始尺寸和缩放因子
        img_type_str = 'rgb' if img_type == 0 else 'ir'
        
        # 确保orig_shapes和scale_factors存在
        if not hasattr(self, 'orig_shapes') or not hasattr(self, 'scale_factors'):
            self.orig_shapes = {
                'rgb': [orig_img.shape[:2] for orig_img in orig_imgs],
                'ir': [orig_img.shape[:2] for orig_img in orig_imgs]
            }
            self.scale_factors = {
                'rgb': [(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1]) 
                       for shape in self.orig_shapes['rgb']],
                'ir': [(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1]) 
                      for shape in self.orig_shapes['ir']]
            }

        results = []
        for i, (pred, orig_img, img_path) in enumerate(zip(preds, orig_imgs, self.batch[img_type][0])):
            # 缩放预测框到原始图像尺寸
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            # 创建Results对象
            result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            
            # 如果需要，可以根据原始尺寸进一步调整预测框
            if hasattr(result, 'boxes') and result.boxes is not None:
                orig_shape = self.orig_shapes[img_type_str][i]
                scale_factors = self.scale_factors[img_type_str][i]
                
                # 调整预测框坐标
                boxes = result.boxes
                if boxes.xyxy is not None:
                    boxes.xyxy[:, [0, 2]] = boxes.xyxy[:, [0, 2]] / scale_factors[1]  # 调整x坐标
                    boxes.xyxy[:, [1, 3]] = boxes.xyxy[:, [1, 3]] / scale_factors[0]  # 调整y坐标
                
                # 更新其他相关属性
                if hasattr(boxes, 'xywh'):
                    boxes.xywh[:, [0, 2]] = boxes.xywh[:, [0, 2]] / scale_factors[1]
                    boxes.xywh[:, [1, 3]] = boxes.xywh[:, [1, 3]] / scale_factors[0]
                
                if hasattr(boxes, 'xyxyn'):
                    boxes.xyxyn[:, [0, 2]] = boxes.xyxyn[:, [0, 2]] * orig_shape[1]
                    boxes.xyxyn[:, [1, 3]] = boxes.xyxyn[:, [1, 3]] * orig_shape[0]
            
            results.append(result)
        
        return results