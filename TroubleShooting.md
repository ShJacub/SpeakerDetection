# TroubleShooting  

1. After restart docker container cuda is not available in torch  
  Settings:  
    Cases:  

      1:
        System:  
          Nvidia Driver 535.86.10  
          Cuda 12.2  
        Docker container:  
          docker image nvidia/cuda:12.4.1-devel-ubuntu22.04  
            Ubuntu22.04  
            Cuda 12.4.1  
      2:
        System:  
          Nvidia Driver 535.86.10  
          Cuda 12.2  
        Docker container:  
          docker image nvidia/cuda:12.4.1-devel-ubuntu20.04  
            Ubuntu20.04  
            Cuda 12.4.1  

  There is no problem with docker images based on ubuntu{22.04/20.04} with cuda 12.1 and 11.8 inside docker container.
  
  There is no problem with docker image nvidia/cuda:12.4.1-devel-ubuntu{22.04/20.04} when System has Nvidia Driver 550.54.14 and cuda 12.4  
  
2. During installation VideoProcessingFramework Python3 can not be found when non-default python3 is installed  
  System:  
    ubuntu20.04  
    python3.9  
  Solution:  
    run apt-get install python3.9-dev -y instead of apt-get install python3-dev -y  