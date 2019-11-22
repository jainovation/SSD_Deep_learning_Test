Docker 설치 방법
================
1. Installing Docker Engine
---------------------------------

>### $ sudo wget -qO- https://get.docker.com/ | sh
>### $ sudo usermod -aG docker $USER
>### $ docker run hello-world

2. Installing Docker Compose
----------------------------------

>### $ sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
>### $ sudo chmod +x /usr/local/bin/docker-compose

>### $ sudo docker-compose --version
docker-compose version 1.21.2, build a133471

nvidia docker 를 사용할 것이기 때문에 추가적인 작업이 필요합니다.

일딴 NVIDIA drivers 를 설치하여 준다. 아래는 Prerequisites이다.  fermi가 아마 4세대인걸로 알고있는데 이 이후의 gpu와 ubuntu 16.04 위 방법으로 도커를 설치하였다면 NVIDIA drivers  만 설치하면 된다.

1. GNU/Linux x86_64 with kernel version > 3.10
2. Docker >= 1.12
3. NVIDIA GPU with Architecture > Fermi (2.1)
4. NVIDIA drivers ~= 361.93 (untested on older versions)

>### $curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
>### $sudo apt-get update

>### $sudo apt-get install nvidia-docker2 
>### $sudo pkill -SIGHUP dockerd

>### docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

<사용예제>
>### $nvidia-docker run -it --name mycuda  --runtime=nvidia  --ipc=host  -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix  -v /data:/data  -p 80:80 -p 443:443 -p 8097:8097 -p 8080:8080 -p 6000:6000 --restart="always" nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04  bash
