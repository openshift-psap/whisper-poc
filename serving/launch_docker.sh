GPUS='"device=0"'
docker run --shm-size=2g -it -d -p 8001:8001 --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus $GPUS \
    --name trt trt-whisper:latest