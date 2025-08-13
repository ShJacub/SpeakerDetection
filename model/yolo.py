import os

import torch
import torchvision

from ultralytics import YOLO
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils import ops

from utils.utils import load_toml_config

def export(weight_path, imgsz, half, batch):

    # Load the YOLOv8 model
    model = YOLO(weight_path)
    model.export(format="engine", imgsz=imgsz, half=half, batch=batch)  # creates 'yolov8n.engine'
    
class Yolov8:
    def __init__(self, config):

        config = load_toml_config(config) if isinstance(config, str) else config

        engine_path = os.path.splitext(config["weight_path"])[0] + ".engine"
        if config["trt_compile"]:
            export(config["weight_path"], config["imgsz"], config["half"], config["batch"])
        self.model = YOLO(engine_path, task="pose")

        self.imgsz = config["imgsz"]
        self.iou = config["iou"]
        self.half = config["half"]
        self.conf = config["conf"]
        self.device = config["device"]

        self.transform = torchvision.transforms.Resize(self.imgsz)
        
    def postprocess(self, preds, shape, orig_shape):
        preds = ops.non_max_suppression(
            preds,
            self.model.predictor.args.conf,
            self.model.predictor.args.iou,
            agnostic=self.model.predictor.args.agnostic_nms,
            max_det=self.model.predictor.args.max_det,
            classes=self.model.predictor.args.classes,
            nc=len(self.model.predictor.model.names),
        )
        
        for pred in preds:
            pred[:, :4] = ops.scale_boxes(shape, pred[:, :4], orig_shape).round()
        
        preds = torch.stack(preds, dim=0)
        preds = preds[..., :6]
            
        return preds
        

    @smart_inference_mode()
    def fast_predict(self, img):

        is_tensor = isinstance(img, torch.Tensor)
        
        im = self.transform(img) if is_tensor else [img]
    
        im = self.model.predictor.preprocess(im)

        preds = self.model.predictor.inference(im)
        
        results = self.postprocess(preds, im.shape[2:], img.shape[2:] if is_tensor else img.shape[:2])

        return results

    def predict(self, img):
        out = self.fast_predict(img)
        return out

    def warm_up(self, img=None):
        if img is None:
            img = torch.zeros((224, 224, 3), dtype=torch.float32, device=self.device)
        img = self.transform(img)
        self.model(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, half=self.half, device=self.device)