import os
from easydict import EasyDict
from shutil import rmtree
import datetime

from natsort import natsorted
import cv2
import numpy as np
import torch
import kornia
import torchvision.transforms as TT

from .ffmpeg import get_video_duration, get_video_lenght, make_video_segments


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def crop_thumbnail(image, bounding_box, padding=1, size=100):

    # infos in original image
    w, h = image.shape[1], image.shape[0]
    x1, y1, x2, y2 = [min(max(int(x), 0), max_v) for x, max_v in zip(bounding_box, [w, h, w, h])]

    img = image.copy()

    output = img[y1:y2, x1:x2]
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)

    new_bbox = None

    return output, new_bbox


def torch_crop_thumbnail(image, bounding_box, padding=1, size=100):

    # infos in original image
    h, w = image.shape[2:]
    x1, y1, x2, y2 = [min(max(int(x), 0), max_v) for x, max_v in zip(bounding_box, [w, h, w, h])]

    output = image[:, :, y1:y2, x1:x2]
    output = TT.functional.resize(output, (size, size), interpolation=TT.InterpolationMode.BILINEAR)
    new_bbox = None

    return output, new_bbox


def side_padding(input_tensor, amount, side):
    result = torch.zeros_like(input_tensor)[0]
    result = torch.stack([result for i in range(amount)], dim=0)
    if side == 'left':
        return torch.cat((result, input_tensor), dim=0)
    else:
        return torch.cat((input_tensor, result), dim=0)


def padding(input_tensor, start_input, end_input, start, end):
    result = input_tensor[(max(start_input, start) - start_input):(min(end_input, end) - start_input + 1), :]
    # print(result.shape[0])
    if start_input > start:
        result = side_padding(result, start_input - start, 'left')
    if end_input < end:
        result = side_padding(result, end - end_input, 'right')
    return result


def draw(img, bboxes, color):
    # bboxes = bboxes[..., :4].to(torch.int64).unsqueeze(0)
    bboxes = bboxes[..., :4].to(torch.int64)
    img = kornia.utils.draw_rectangle(
            img,
            bboxes,
            color=torch.tensor(color, device='cuda:0'),
            fill=False

        )
    return img

def box_for_draw(boxes):
    boxes = torch.tensor(boxes)
    boxes = boxes.reshape(1, *boxes.shape) if len(boxes) else torch.empty(1, 0, 4)
    return boxes


import yaml
def load_cfg(path_to_cfg):
    with open(path_to_cfg, "r") as fopen:
        return EasyDict(yaml.safe_load(fopen))
    
import toml
def load_toml_config(config_path):
    with open(config_path, "r") as fopen:
        config = toml.load(fopen)
    return config
    
def get_time(t):
    h, m, s = t // 360, t % 360 // 60, t % 60
    return str(datetime.time(hour=h, minute=m, second=s))


def delete_dir(split_dir):
    if os.path.exists(split_dir):
        rmtree(split_dir)

def separate_paths(video_paths):
    sep_video_paths = {}
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sh = video_name.split('_')[1]
        if not sh in sep_video_paths:
            sep_video_paths[sh] = []
        sep_video_paths[sh].append(video_path)

    return sep_video_paths

def paths2sh_idx(sh_to_paths):
    path_to_sh = {}
    for sh, video_paths in sh_to_paths.items():
        for idx, video_path in enumerate(video_paths):
            path_to_sh[video_path] = [sh, idx]
    return path_to_sh

def cumul_values(xl):
    new_xl = [0]
    for value in xl:
        new_xl.append(new_xl[-1] + value)
    return new_xl[1:]

def get_start_end_video_indexes(cum_values):
    cum_values = [0] + cum_values
    return np.array([cum_values[i:i + 2] for i, _ in enumerate(cum_values[:-1])], dtype=np.int64)

def move_right(sh_list, cur_i):
    cur_i += 1
    cur_i = cur_i if cur_i < len(sh_list) else 0
    return cur_i, sh_list[cur_i]

def del_duplicated_idxs(indexes):
    set_idxs = set()
    new_indexes = []
    for i, adj_indexes in enumerate(indexes):
        new_indexes.append({})
        for sh in natsorted(adj_indexes.keys()):
            idxs = [idx for idx in adj_indexes[sh] if not idx in set_idxs]
            if len(idxs):
                new_indexes[-1][sh] = idxs
            set_idxs = set_idxs.union(set(adj_indexes[sh]))
            
    new_indexes = new_indexes if len(new_indexes[-1]) else new_indexes[:-1]
            
    return new_indexes

def get_main_frames_indexes(vid_indexes):
    mf_indexes = []
    for adj_vid_indexes in vid_indexes:
        sh_list = natsorted(adj_vid_indexes.keys())
        mf_indexes.append({})
        for i, sh in enumerate(sh_list):
            i_f, sh_f = move_right(sh_list, i)
            i_l, sh_l = move_right(sh_list, i_f)
            f_ch_idx = 0 if i <= i_f else 1
            l_ch_idx = 0 if i < i_l else 1
            f = adj_vid_indexes[sh_f][f_ch_idx]
            l = adj_vid_indexes[sh_l][l_ch_idx]
            mf_indexes[-1][sh] = list(range(f, l))
    
    mf_indexes = del_duplicated_idxs(mf_indexes)

    # First video interval indexes correcting (indexes before main indexes + main indexes)
    mf_indexes[0][sh_list[0]] = list(range(mf_indexes[0][sh_list[0]][-1] + 1))

    # Getting relative indexes
    mf_indexes = [{sh : [x - vid_indexes[i][sh][0] for x in mf_vid_indexes] \
                        for sh, mf_vid_indexes in adj_mf_indexes.items()} \
                        for i, adj_mf_indexes in enumerate(mf_indexes)]
    return mf_indexes

def swap_idx_keys(data):
    shs = data.keys()
    main_p_name = min(shs)
    return [{sh : data[sh][idx] for sh in shs if idx < len(data[sh])} \
             for idx, _ in enumerate(data[main_p_name])]


def split_video(video_path, split_dir="split_videos"):

    window = 10
    segment_window = 3 * window
    segment_time = get_time(segment_window)

    os.makedirs(split_dir, exist_ok=True)

    video_duration = get_video_duration(video_path)

    for i in range(3):

        start_time = i * window
        if start_time >= video_duration:
            continue

        start_time = get_time(i * window)

        dst_path = f"{split_dir}{os.sep}output%03d_{i}.mkv"
        make_video_segments(start_time, video_path, segment_time, dst_path)


def merge_results(faces, video_paths):

    # Distributing videos into segments
    sh_to_paths = separate_paths(video_paths)
    # From (key -> value) to (value -> key)
    path_to_sh_idx = paths2sh_idx(sh_to_paths)
    # Getting video lengths in frames
    vid_p = {sh : [get_video_lenght(video_path) for video_path in sh_video_paths] \
                 for sh, sh_video_paths in sh_to_paths.items()}
    # Getting segments lengths in frames
    sh_lens = {sh : sum(sh_vid_p) for sh, sh_vid_p in vid_p.items()}
    # Getting shifts of each segment relative to main segment
    main_p_name = min(sh_lens.keys())
    shifts = {sh : sh_lens[main_p_name] - sh_lens[sh] for sh in sh_lens}
    # Cumulative values
    vid_p = {sh : cumul_values(sh_vid_p) for sh, sh_vid_p in vid_p.items()}
    # Getting relative indexes(start, end) of videos
    vid_p = {sh : get_start_end_video_indexes(sh_vid_p) for sh, sh_vid_p in vid_p.items()}
    # Getting absolute indexes(start, end) of videos
    vid_p = {sh : shifts[sh] + vid_p[sh] for sh in vid_p}
    # Swap key : list -> idx : dict
    vid_p = swap_idx_keys(vid_p)
    # Getting absolute main frames indexes
    vid_p = get_main_frames_indexes(vid_p)

    merge_faces = []
    for video_faces, video_path in zip(faces, video_paths):
        sh, idx = path_to_sh_idx[video_path]
        
        if idx >= len(vid_p) or not sh in vid_p[idx]:
            continue
        
        fidxs = vid_p[idx][sh]
        
        for fidx in fidxs:
            merge_faces.append(video_faces[fidx])

    num_frames = sh_lens[natsorted(list(sh_lens.keys()))[0]]
    assert num_frames == len(merge_faces), f"Numger of frames({num_frames}) does not match to number of merge_faces({len(merge_faces)})"

    return merge_faces

def set_folder_cfg(cfg):
    cfg.split_dir = os.path.join(cfg.tmp_dir, cfg.split_folder)
    cfg.mp4_split_dir = f"{cfg.split_dir}_mp4"
    return cfg

def create_dirs(main_cfg):
    # tmp dirs
    os.makedirs(main_cfg.split_dir)
    os.makedirs(main_cfg.mp4_split_dir)
    # result dir
    os.makedirs(main_cfg.result_dir)