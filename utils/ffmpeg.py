import os
import subprocess

import tqdm
from natsort import natsorted


def get_video_duration(video_path):
    res = subprocess.run(f'ffprobe -i {video_path} -show_entries format=duration -v quiet -of csv="p=0"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(res.stdout)


def get_video_lenght(video_path):
    res = subprocess.run(f'ffprobe -i {video_path} -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -v quiet -of csv="p=0"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(res.stdout)

    
def video_framerate_reset(video_path, fr=25, save_dir=None):
    new_video_path, ext = os.path.splitext(video_path)
    new_video_path = f"{video_path}_{fr}" + ext
    new_video_path = os.path.join(save_dir, os.path.basename(new_video_path)) if save_dir else new_video_path
    command = (f"ffmpeg -y -i {video_path} -qscale:v 2 -async 1 -r {fr} {new_video_path} -loglevel panic")
    subprocess.call(command, shell=True, stdout=True)
    return new_video_path


def mkv2mp4(src_path, dst_path):
    assert os.path.splitext(dst_path)[1] == ".mp4", "Wrong extension of video in dst_path"
    # command = (f"ffmpeg -y -i {src_path} -c:v copy -c:a copy {dst_path}")
    command = (f"ffmpeg -y -i {src_path} -c:v copy -c:a copy -r 25 {dst_path}")
    subprocess.call(command, shell=True, stdout=True)


def extract_audio(video_path, nDataLoaderThread=10):
    assert os.path.splitext(video_path)[1] == ".mp4", "Wrong extension of video_path"
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (video_path, nDataLoaderThread, audio_path))
    subprocess.call(command, shell=True, stdout=None)
    return audio_path


def mkvs2mp4s(videos_dir="split_videos", new_video_dir="split_videos_mp4"):
    os.makedirs(new_video_dir, exist_ok=True)
    video_paths = [os.path.join(videos_dir, video_name) for video_name in natsorted(os.listdir(videos_dir))]
    for video_path in tqdm.tqdm(video_paths, desc="converting mkv into mp4"):
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0] + ".mp4"
        new_video_path = os.path.join(new_video_dir, video_name)
        mkv2mp4(video_path, new_video_path)


def make_video_segments(start_time, video_path, segment_time, dst_path):
    command = (f"ffmpeg -y -ss {start_time} -i {video_path} -map_metadata -1 -c:v copy -c:a copy -segment_time {segment_time} -f segment -reset_timestamps 1 {dst_path}")
    subprocess.call(command, shell=True, stdout=True)