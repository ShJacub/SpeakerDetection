import time

import numpy as np
import torch
import mediapipe as mp
from mediapipe import solutions

from .utils import crop_thumbnail

lips = [
        # lipsUpperOuter
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # lipsLowerOuter in reverse order
        375, 321, 405, 314, 17, 84, 181, 91, 146, 61,

        76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306,

        307, 320, 404, 315, 16, 85, 180, 90, 77,

        62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292,

        325, 319, 403, 316, 15, 86, 179, 89, 96,

        #  lipsUpperInner
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # lipsLowerOuter in reverse order
            324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]


def get_lips_landmarks(landmark_lists, H, W):
    lp_landmark_lists = []
    for id in lips:
        if id >= len(landmark_lists):
            lp_landmark_lists.append((-1, -1))
            continue
        landmark = landmark_lists[id]
        # print(landmark)
        landmark_px = solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y, W, H)
        # print(landmark_px)

        if landmark_px:
            lp_landmark_lists.append((landmark.x, landmark.y))
        else:
            lp_landmark_lists.append((-1, -1))
            
    return lp_landmark_lists

def det_kpts(image, bbox, kpts_detector):
    # crop face
    H = 128
    faceCrop, _ = crop_thumbnail(
        image, bbox, padding=0.775, size=H)
    
    faceCrop = mp.Image(image_format=mp.ImageFormat.SRGB, data=faceCrop)
    
    detection_result = kpts_detector.detect(faceCrop)
    
    landmark_lists = []
    for i in range(len(detection_result.face_landmarks)):
        landmark_lists.extend(detection_result.face_landmarks[i])
        
    lp_landmark_lists = get_lips_landmarks(landmark_lists, H, H)
    
    return lp_landmark_lists


def one_pass_det_inference(img, DET):

    torch.cuda.synchronize()
    t0 = time.time()

    bboxes = DET.predict(img / 255)[0].cpu().numpy()[..., :5]

    torch.cuda.synchronize()
    t0 = time.time()- t0

    img_dets = [
        {'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]} \
        for bbox in bboxes
    ]

    return img_dets

def make_inference_use_landmarks(use_landmarks):
    def actual_make_inference_use_landmarks(func):
        def wrapper(*args, **kwargs):
            img_dets = func(*args)

            image = args[[0]].type(torch.uint8).squeeze(0).permute(1, 2, 0).flip(dims=[2]).cpu().numpy()
            image = np.ascontiguousarray(image)

            [img_det.update({'landmarks' : det_kpts(image, img_det['bbox'], kwargs['kpts_detector'])}) for img_det in img_dets]
            
            return img_dets
        return wrapper if use_landmarks else func
    return actual_make_inference_use_landmarks