export DEBIAN_FRONTEND=noninteractive
apt update
apt install python3 -y
# apt install python3.9 -y
apt install wget -y
wget https://bootstrap.pypa.io/get-pip.py
# wget https://bootstrap.pypa.io/pip/3.8/get-pip.py
apt install python3-distutils -y
# update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# apt-get install python3.9-dev -y
apt-get install python3-dev -y
python3 get-pip.py

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 # has to work. it was used following command last time because. This need to be checked 
# pip install tensorrt==10.1.0 tensorrt-cu12==10.12.0.36

pip install -r speaker_detection/requirements.txt