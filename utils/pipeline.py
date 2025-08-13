import sys
import time
import os
import subprocess
import warnings
from easydict import EasyDict
from collections import defaultdict
import datetime
from shutil import rmtree
import random

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from natsort import natsorted
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as TT
from torio.io import StreamingMediaEncoder as StreamWriter, CodecConfig
from scipy.interpolate import interp1d
from scipy.io import wavfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .video import VideoDecoder
from .utils import (
    bb_intersection_over_union,
    torch_crop_thumbnail,
    padding,
    box_for_draw,
    draw,
    load_cfg
)
from .detector import (
    make_inference_use_landmarks,
    one_pass_det_inference
)
from model import loconet
from model.yolo import Yolov8
from model.torchvggish import vggish_input


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    return sceneList


def det_inference(args, one_pass_det_func, DET, **kwargs):
    # GPU: Face detection, output is the list contains the face location and score in this frame

    # Video Reader
    video = VideoDecoder(video_path=args.videoFilePath).init()
    dataloader = DataLoader(video, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)
    args.number_frames = len(video)
    
    dets = []
    for fidx, sample in enumerate(tqdm.tqdm(dataloader)):
        dets.append(one_pass_det_func(sample["img"], DET, **kwargs))
        [img_dets.update({'frame' : fidx}) for img_dets in dets[-1]]
        
    return dets


def form_tracks(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5     # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            tracks.append(track)
            
    
    return tracks


def form_track_feature(track):
    frameNum = np.array([f['frame'] for f in track])
    bboxes = np.array([np.array(f['bbox']) for f in track])
    frameI = np.arange(frameNum[0], frameNum[-1]+1)
    bboxesI = []
    for ij in range(0, 4):
        interpfn = interp1d(frameNum, bboxes[:, ij])
        bboxesI.append(interpfn(frameI))
    # print(landmarks.shape)
    # for i in range(82):
    #     p_lms = []
    #     for j in range(2):
    #         p_lms.append(interp1d(frameNum, landmarks[:, i, j])(frameI))
    #     print([x.shape for x in p_lms])
    #     landmarksI.append(np.stack(p_lms, axis=1))
    # landmarksI = np.stack(landmarksI, axis=1)
    bboxesI = np.stack(bboxesI, axis=1)
    return {'frame': frameI, 'bbox': bboxesI}


def make_track_use_landmarks(use_landmarks):
    def actual_make_track_use_landmarks(func):
        def wrapper(*args):

            track = func(*args)
            
            frameNum = np.array([f['frame'] for f in args[0]])
            frameI = np.arange(frameNum[0], frameNum[-1]+1)
            landmarks = np.array([f['landmarks'] for f in args[0]])
            landmarksI = np.stack([np.stack([interp1d(frameNum, landmarks[:, i, j])(frameI) for j in range(2)], axis=1) for i in range(82)], axis=1)
            track.update({'landmarks' : landmarksI})
            
            return track
        
        return wrapper if use_landmarks else func
    return actual_make_track_use_landmarks
            

def track_shot(args, sceneFaces, form_track_feature_func):
    # CPU: Face tracking
    tracks = form_tracks(args, sceneFaces)
    feature_tracks = []
    for track in tracks:
        track = form_track_feature_func(track)
        if max(np.mean(track['bbox'][:, 2]-track['bbox'][:, 0]), np.mean(track['bbox'][:, 3]-track['bbox'][:, 1])) > args.minFaceSize:
            feature_tracks.append(track)
    return feature_tracks


def frame_num2tidx_fidx(tracks):
    frame_num_to_tidx_fidx = defaultdict(list)
    for tidx, track in enumerate(tracks):
        for fidx, frame_num in enumerate(track['frame']):
            frame_num_to_tidx_fidx[frame_num].append([tidx, fidx])
    return frame_num_to_tidx_fidx

def prepare_frame_num_info(args, tracks, frame_num_to_tidx_fidx):
    
    frame_num_info = [{'frame': []} for i in range(len(tracks))]
    
    for frame_num in range(args.number_frames):
        if not frame_num in frame_num_to_tidx_fidx:
            continue
        
        for tidx, fidx in frame_num_to_tidx_fidx[frame_num]:
            
            # store information
            frame_num_info[tidx]['frame'].append(frame_num)

    return frame_num_info


def collect_candidates(frame_num_info):
    # loop through each tracked identity
    candidates = []
    for person_id in range(len(frame_num_info)):
        # get list of people that have at least half time with person_id
        candidate = []

        for i in range(len(frame_num_info)):
            if i == person_id:
                continue

            intersect = set(frame_num_info[i]['frame']).intersection(
                set(frame_num_info[person_id]['frame']))
            if len(intersect) >= len(frame_num_info[person_id]['frame']) / 2:
                candidate.append(
                    {'id': i, 'start': frame_num_info[i]['frame'][0], 'end': frame_num_info[i]['frame'][-1]})
        candidates.append(candidate)
        
    return candidates


def prepare_visual(args,
                   tracks,
                   frame_num_to_tidx_fidx,
                   frame_num_info,
                   candidates):
    # prepare inputs
    visual_info = [{'faceCrop': []} for i in range(len(tracks))]

    H = 112
    
    video = VideoDecoder(video_path=args.videoFilePath).init()
    dataloader = DataLoader(video, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)

    for frame_num, sample in enumerate(tqdm.tqdm(dataloader)):
        if not frame_num in frame_num_to_tidx_fidx:
            continue
        
        frame = sample['img']
        
        for tidx, fidx in frame_num_to_tidx_fidx[frame_num]:
            
            track = tracks[tidx]

            # crop face
            faceCrop, _ = torch_crop_thumbnail(
                frame, track['bbox'][fidx], padding=0.775, size=H)
            faceCrop = TT.functional.rgb_to_grayscale(faceCrop)[:, 0]

            # store information
            visual_info[tidx]['faceCrop'].append(faceCrop)
            
    
    for tidx, _ in enumerate(visual_info):
        visual_info[tidx]['faceCrop'] = torch.cat(
            visual_info[tidx]['faceCrop'], dim=0)

    visual_feature = []

    # loop through each tracked identity
    for person_id in range(len(visual_info)):
        # get list of people that have at least half time with person_id
        candidate = candidates[person_id]

        visualFeature = None

        # extract visual input
        if len(candidate) == 0:
            visualFeature = torch.stack([visual_info[person_id]['faceCrop'], visual_info[person_id]
                                        ['faceCrop'], visual_info[person_id]['faceCrop']], dim=0)
        elif len(candidate) == 1:
            context = padding(visual_info[candidate[0]['id']]['faceCrop'], candidate[0]['start'], candidate[0]
                              ['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            visualFeature = torch.stack(
                [visual_info[person_id]['faceCrop'], context, visual_info[person_id]['faceCrop']], dim=0)
            
        else:
            random.shuffle(candidate)
            candidate1 = padding(visual_info[candidate[0]['id']]['faceCrop'], candidate[0]['start'],
                                 candidate[0]['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            candidate2 = padding(visual_info[candidate[-1]['id']]['faceCrop'], candidate[-1]['start'],
                                 candidate[-1]['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            visualFeature = torch.stack(
                [visual_info[person_id]['faceCrop'], candidate1, candidate2], dim=0)

        visual_feature.append(visualFeature.unsqueeze(0))

    return visual_feature


def prepare_landmarks(args,
                      tracks,
                      frame_num_to_tidx_fidx,
                      frame_num_info,
                      candidates):
    visual_info = [{'landmarks' : []} for i in range(len(tracks))]
    
    for frame_num in range(args.number_frames):
        if not frame_num in frame_num_to_tidx_fidx:
            continue
        
        for tidx, fidx in frame_num_to_tidx_fidx[frame_num]:
            
            track = tracks[tidx]

            landmarks = torch.from_numpy(track['landmarks'][fidx])

            visual_info[tidx]['landmarks'].append(landmarks)
            
        visual_info[tidx]['landmarks'] = torch.stack(
            visual_info[tidx]['landmarks'], dim=0)
        
    landmarks = []

    # loop through each tracked identity
    for person_id in range(len(visual_info)):
        # get list of people that have at least half time with person_id
        candidate = candidates[person_id]

        landMarks = None

        # extract visual input
        if len(candidate) == 0:
            landMarks = torch.stack([visual_info[person_id]['landmarks'], visual_info[person_id]['landmarks'],
                                    visual_info[person_id]['landmarks']], dim=0)
        elif len(candidate) == 1:
            context = padding(visual_info[candidate[0]['id']]['landmarks'], candidate[0]['start'], candidate[0]
                            ['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            landMarks = torch.stack(
                [visual_info[person_id]['landmarks'], context, visual_info[person_id]['landmarks']], dim=0)
        else:
            candidate1 = padding(visual_info[candidate[0]['id']]['landmarks'], candidate[0]['start'],
                                candidate[0]['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            candidate2 = padding(visual_info[candidate[-1]['id']]['landmarks'], candidate[-1]['start'],
                                candidate[-1]['end'], frame_num_info[person_id]['frame'][0], frame_num_info[person_id]['frame'][-1])
            landMarks = torch.stack(
                [visual_info[person_id]['landmarks'], candidate1, candidate2], dim=0)

        landmarks.append(landMarks.unsqueeze(0))

    return landmarks
        

def prepare_audio(args,
                  tracks,
                  frame_num_info,
                  visual_feature):
    
    audio_feature = []
    for person_id in range(len(frame_num_info)):
    
        # extract audio
        # audioTmp = os.path.join(args.tmp_dir, f'audio{person_id:06d}.wav')
        audioTmp = os.path.join(args.tmp_dir, f'audio.wav')
        audioStart = tracks[person_id]['frame'][0] / 25
        audioEnd = tracks[person_id]['frame'][-1] / 25
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                    (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
        subprocess.call(command, shell=True, stdout=False)
        sr, wav_data = wavfile.read(audioTmp)
        assert audioStart < audioEnd
        audioFeature = vggish_input.waveform_to_examples(
            wav_data, sr, visual_feature[person_id].shape[2], 25, False)

        # audio feature has shape (B, C, 4*T, M)
        audioFeature = torch.from_numpy(audioFeature).unsqueeze(0).unsqueeze(0)
        audio_feature.append(audioFeature)

    return audio_feature

def prepare_visual_landmarks(*args):
    return {"visual_feature" : prepare_visual(*args)}

def make_prepare_use_landmarks(use_landmarks):
    def actual_make_prepare_use_landmarks(func):
        def wrapper(*args):
            features = func(*args)
            features.update({'landmarks' : prepare_landmarks(*args)})
            return features
        return wrapper if use_landmarks else func
    return actual_make_prepare_use_landmarks


def prepare_input(args, tracks, prepare_func):
    frame_num_to_tidx_fidx = frame_num2tidx_fidx(tracks)
    frame_num_info = prepare_frame_num_info(args, tracks, frame_num_to_tidx_fidx)
    candidates = collect_candidates(frame_num_info)
    v_l_feature = prepare_func(args, tracks, frame_num_to_tidx_fidx, frame_num_info, candidates)
    audio_feature = prepare_audio(args, tracks, frame_num_info, v_l_feature["visual_feature"])
    return audio_feature, v_l_feature


def inference(model, lenTracks, audio_feature, visual_feature, **kwargs):

    with torch.no_grad():
        result = [None for i in range(lenTracks)]

        # TODO: make a forward pass to make prediction
        for i in tqdm.tqdm(range(lenTracks)):
            visual = visual_feature[i]
            visualFeature = visual.to(dtype=torch.float, device='cuda')

            # get audio feature
            audioFeature = audio_feature[i].to(dtype=torch.float, device='cuda')

            kwargs_i = {key : value[i].to(dtype=torch.float, device='cuda') \
                        for key, value in kwargs.items()}

            # run frontend part of the model
            predScore = model.model.forward(audioFeature, visualFeature, **kwargs_i)
            # print(predScore)
            # print(sum(predScore < 0))
            ###### Custom
            predScore = predScore[:, 1].detach().cpu().numpy()
            ######
            result[i] = predScore

    return result


def vpf_visualization(faces, video_path, audio_path, main_cfg):
    
    vid_save_path = os.path.join(main_cfg.tmp_dir, "out.mp4")
    vid_au_save_path = os.path.join(main_cfg.result_dir, "out.mp4")

    # Video Reader
    video = VideoDecoder(video_path=video_path).init()
    dataloader = DataLoader(video, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)

    fps = video.get_fps()
    width, height = video.get_img_size()
    
    # Init stream writer:
    video_streamer = StreamWriter(dst=vid_save_path)

    encoder_option={
                        # "preset": "losslesshp",
                        "preset" : "fast",
                        # "tune" : "lossless",
                        # "profile" : "high",
                        "delay" : "0",
                        # "threads" : "5",
                        # "thread_type" : "slice",
                        # "zerolatency" : "1"
    }
    codec_config = CodecConfig()
    codec_config.bit_rate = 5000000
    video_streamer.add_video_stream(frame_rate=fps, height=height, width=width, encoder_option=encoder_option,
                                    encoder='h264_nvenc', hw_accel='cuda:0', encoder_format="rgb0",
                                    )

    video_streamer.open()

    for fidx, sample in enumerate(tqdm.tqdm(dataloader)):

        img = sample["img"].type(torch.uint8)

        # if fidx in faces:
        if fidx > 0 and fidx < len(faces):
            # print(f"Here : {fidx}")
            d_faces = [face for face in faces[fidx] if face['score'] >= 0]
            sp_faces = [face['bbox'][:4] for face in d_faces if face['score'] >= 0.5]
            non_faces = [face['bbox'][:4] for face in d_faces if face['score'] < 0.5]
            sp_faces = box_for_draw(sp_faces)
            non_faces = box_for_draw(non_faces)
            img = draw(img, sp_faces, (0, 255, 0))
            img = draw(img, non_faces, (0, 0, 255))

        video_streamer.write_video_chunk(0, img)

    video_streamer.close()

    command = (f"ffmpeg -y -i {vid_save_path} -i {audio_path} -c:v copy -c:a copy {vid_au_save_path}")
    subprocess.call(command, shell=True, stdout=True)

def get_classified_faces(args, pred, tracks):

    faces = [[] for i in range(args.number_frames)]
    for tidx, track in enumerate(tracks):
        score = pred[tidx]
        for fidx, frame in enumerate(track['frame'].tolist()):
            # s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            # s = np.mean(s)
            faces[frame].append(
                {'track': tidx, 'score': score[fidx], 'bbox': track['bbox'][fidx]})

    return faces    


def one_split_part_process(video_path, main_cfg, one_pass_det_func, DET, form_track_feature_func, prepare_func, model, **kwargs):
    
    assert os.path.splitext(video_path)[1] == ".mp4", "Video format must be mp4"

    # Form args
    args = EasyDict({"videoFilePath" : video_path})
    args.update(main_cfg["PROCESS_PARAMS"])
    args.tmp_dir = main_cfg["tmp_dir"]

    warnings.filterwarnings("ignore")

    # Extract audio
    args.audioFilePath = os.path.join(args.tmp_dir, 'audio.mp3')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (video_path, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Extract the audio and save in %s \r\n" % (args.audioFilePath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Scene detection is done")


    # Face detection for the video frames
    faces = det_inference(args, one_pass_det_func, DET, **kwargs)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face detection is done \r\n")

    # Face tracking
    allTracks = []
    for shot in scene:
        # Discard the shot frames less than minTrack frames
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
            allTracks.extend(track_shot(
                args, faces[shot[0].frame_num:shot[1].frame_num], form_track_feature_func))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                     " Face track and detected %d tracks \r\n" % len(allTracks))

    audio_feature, v_l_feature = prepare_input(args, allTracks, prepare_func)

    result = inference(model, len(allTracks), audio_feature, **v_l_feature)

    faces = get_classified_faces(args, result, allTracks)

    return faces


def run_one_videos(main_cfg):
    videos_dir = main_cfg.mp4_split_dir
    video_paths = [os.path.join(videos_dir, video_name) for video_name in natsorted(os.listdir(videos_dir))]

    # Init Face Detector
    config_path = "configs/yolo.cfg"
    DET = Yolov8(config_path)
    # Warm up
    DET.warm_up()

    # initialize speaker detector model
    cfg = load_cfg("configs/asd.yaml")
    model = loconet(cfg)
    model.loadParameters('LoCoNet_LASER.model' if cfg.USE_LASER_LOCONET else 'weights/loconet_ava_best.model')
    model = model.to(device='cuda')
    model.eval()

    # Landmark detector
    kwargs = {}
    if cfg.USE_LASER_LOCONET:
        base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task', delegate = python.BaseOptions.Delegate.GPU)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                        output_face_blendshapes=True,
                                                        output_facial_transformation_matrixes=True,
                                                        num_faces=1)
        kwargs["kpts_detector"] = vision.FaceLandmarker.create_from_options(options)

    one_pass_det_func = make_inference_use_landmarks(cfg.USE_LASER_LOCONET)(one_pass_det_inference)
    prepare_func = make_prepare_use_landmarks(cfg.USE_LASER_LOCONET)(prepare_visual_landmarks)
    form_track_feature_func = make_track_use_landmarks(cfg.USE_LASER_LOCONET)(form_track_feature)
    
    faces = []
    for video_path in tqdm.tqdm(video_paths, desc="running on videos ..."):
        video_faces = one_split_part_process(video_path, main_cfg, one_pass_det_func, DET, form_track_feature_func, prepare_func, model, **kwargs)
        faces.append(video_faces)
    return faces, video_paths