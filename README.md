
This repository aims to guide the deployment and
testing of multimodal whisper models on both
vLLM and TensorRT.

## Container builds

Container images are published in
[quay.io](https://quay.io/repository/psap/whisper-poc?tab=tags).

- TensorRT-LLM `quay.io/psap/whisper-poc:latest-trt`.
- vLLM `quay.io/psap/whisper-poc:latest-vllm`.

### Building TensorRT container

```
podman build -f Containerfile.trt -t quay.io/psap/whisper-poc:latest-trt .
podman push quay.io/psap/whisper-poc:latest-trt
```

### Building vLLM container

```
podman build -f Containerfile.vllm -t quay.io/psap/whisper-poc:latest-vllm .
podman push quay.io/psap/whisper-poc:latest-vllm
```

## Deploying

To deploy the containers in a OCP cluster run (from the repository root folder):

- Run initial steps `./00_pre.sh`.
- Deploy a TensorRT-LLM pod with `./01_pod.trt.sh` or
- Deploy a vLLM pod with `./01_pod.vllm.sh`.

## Testing

### vLLM

- Connect to the vLLM container with and run the evaluation script with `python /workspace/scripts/run_vllm.py`.

```
oc exec -n my-whisper-runtime -it vllm-standalone -- /bin/bash
```

- Run the script directly:

```
oc exec -n my-whisper-runtime -it vllm-standalone -- /bin/bash -c "python /workspace/scripts/run_vllm.py"
```

The current output should look like:

```
.
.

Elapsed time: 789.2372903823853
Total audio seconds processed: 49507.556
Seconds transcribed / sec: 62.72835382121085
Requests per second: 4.217996337181559 for 3329
.
.
.
```


### TensorRT-LLM
