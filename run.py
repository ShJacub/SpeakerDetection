import os
import warnings


from utils.utils import (
    load_cfg,
    set_folder_cfg,
    delete_dir,
    create_dirs,
    split_video,
    merge_results,
)
from utils.ffmpeg import (
    video_framerate_reset,
    mkvs2mp4s,
    mkv2mp4,
    extract_audio,
)
from utils.pipeline import (
    run_one_videos,
    vpf_visualization
)

warnings.filterwarnings("ignore")
    

def main():
    main_cfg = "configs/main.yaml"
    
    main_cfg = load_cfg(main_cfg)
    main_cfg = set_folder_cfg(main_cfg)
    
    # Remove dirs
    delete_dir(main_cfg.tmp_dir)
    delete_dir(main_cfg.result_dir)
    
    create_dirs(main_cfg)
    
    # Create temporal_dirs
    
    video_path = video_framerate_reset(main_cfg.video_path, fr=main_cfg.PROCESS_PARAMS.fps, save_dir=main_cfg.tmp_dir)
    # input("1")
    split_video(video_path, split_dir=main_cfg.split_dir)
    # input("2")
    mkvs2mp4s(videos_dir=main_cfg.split_dir, new_video_dir=main_cfg.mp4_split_dir)
    # input("3")
    faces, video_paths = run_one_videos(main_cfg)
    # input("4")
    faces = merge_results(faces, video_paths)
    # input("5")
    mp4_path = os.path.join(main_cfg.tmp_dir, os.path.splitext(os.path.basename(video_path))[0] + ".mp4")
    mkv2mp4(video_path, mp4_path)
    # input("6")
    audio_path = extract_audio(mp4_path)
    # input("7")
    vpf_visualization(faces, mp4_path, audio_path, main_cfg)
    # input("8")

    delete_dir(main_cfg.tmp_dir)


if __name__ == '__main__':
    main()
