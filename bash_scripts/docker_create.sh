docker create -it --ipc=host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video --runtime=nvidia -v /home/jacob/data:/data -v /home/jacob/WorkSpace/to_git/speaker_detection:/speaker_detection --name=asd nvidia/cuda:12.1.0-devel-ubuntu22.04 /bin/bash &&
docker start asd &&
docker exec -it asd /bin/bash