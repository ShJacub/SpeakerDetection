export DEBIAN_FRONTEND=noninteractive LANG=C TZ=UTC
export CMAKE_POLICY_VERSION_MINIMUM=3.5

apt-get update ;\
apt-get upgrade -y ;\
apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    wget \
    python3-dev \
    python3-pip \
    libgl1 \
    unzip \
    git \
    cmake \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev \
    linux-headers-generic \
    dkms

# install some libs for ffmpeg
apt-get update ;\
apt-get install -y \
    pkg-config checkinstall libfaac-dev \
    libgpac-dev ladspa-sdk-dev libunistring-dev libbz2-dev  libjack-jackd2-dev \
    libmp3lame-dev libsdl2-dev libopencore-amrnb-dev libopencore-amrwb-dev \
    libvpx-dev libx264-dev libx265-dev libxvidcore-dev libopenal-dev libopus-dev \
    libsdl1.2-dev libtheora-dev libva-dev libvdpau-dev libvorbis-dev libx11-dev \
    libxfixes-dev texi2html yasm zlib1g-dev gobjc++ x265 libnuma-dev libnuma1 \
    libc6 libc6-dev libtool libssl-dev

mkdir /builds


# install nv-codec-headers:
git clone -b sdk/11.0 https://github.com/FFmpeg/nv-codec-headers.git /builds/nv-codec-headers/
cd /builds/nv-codec-headers && make install

# configure and build ffmpeg from sources:
git clone --depth 1 -b release/6.0 https://git.ffmpeg.org/ffmpeg.git /builds/ffmpeg/
cd /builds/ffmpeg
./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --enable-cuvid \
            --extra-cflags=-I/usr/local/cuda/include \
            --extra-ldflags=-L/usr/local/cuda/lib64 \
            --disable-static --enable-shared \
            --enable-gpl --enable-version3 \
            --enable-libmp3lame --enable-libvpx --enable-libopus \
            --enable-opencl --enable-libxcb --enable-openssl \
            --enable-nvenc --enable-vaapi \
            --enable-vdpau --enable-ffplay --enable-ffprobe \
            --enable-libxvid \
            --enable-libx264 --enable-openal --enable-libx265
make -j && make install
ldconfig # in order not to restart run this command. Some so-files can't be found. This is not necessary if you make in Dockerfile for docker image creation

# install VideoProcessingFramework:
git clone --depth=1 -b set_torch_version https://github.com/ShJacub/VideoProcessingFramework.git /builds/VideoProcessingFramework/
cd /builds/VideoProcessingFramework/ && python3 setup_torch_version.py
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cd /builds/VideoProcessingFramework/ && pip3 install .
cd /builds/VideoProcessingFramework/ && pip3 install src/PytorchNvCodec
