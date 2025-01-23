
This repository aims to guide the deployment and
testing of multimodal whisper models on both
vLLM and TensorRT.

## Container builds

Container images are published in
[quay.io](https://quay.io/repository/psap/whisper-poc?tab=tags).

- TensorRT-LLM `quay.io/psap/whisper-poc:latest-trt`.
- vLLM `quay.io/psap/whisper-poc:latest-vllm`.

### Building TensorRT container

From the repository root folder:

```
cd containers
podman build -f Containerfile.trt -t quay.io/psap/whisper-poc:latest-trt .
podman push quay.io/psap/whisper-poc:latest-trt
```

### Building vLLM container

From the repository root folder:

```
cd containers
podman build -f Containerfile.vllm -t quay.io/psap/whisper-poc:latest-vllm .
podman push quay.io/psap/whisper-poc:latest-vllm
```

## Deploying

To deploy the containers in a OCP cluster run (from the repository root folder):

- Go into the containers folder `cd containers`.
- Run initial steps `./00_pre.sh`.
- Deploy a TensorRT-LLM pod with `./01_pod.trt.sh` or
- Deploy a vLLM pod with `./01_pod.vllm.sh`.

## Testing

### vLLM

Connect to the vLLM container with and run the evaluation script with `python /workspace/scripts/run_vllm.py`.

```
oc exec -n my-whisper-runtime -it vllm-standalone -- /bin/bash
```

Run the script directly:

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

Connect to the TensorRT-LLM container with 

```
oc exec -n my-whisper-runtime -it trt-standalone -- /bin/bash
```

In the container, build the model:

```
bash scripts/trt-build-whisper.sh
```

And start the Triton inference server:

```
source ~/scripts/trt-whisper-vars.sh
cd ~/tensorrtllm_backend
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=model_repo_whisper/ --tensorrt_llm_model_name tensorrt_llm,whisper_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000
```

Alternatively, to do offline inference (don't need to run triton server for this):

```
source ~/scripts/trt-whisper-vars.sh
cd ~/tensorrtllm_backend/tensort_llm/examples/whisper
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3 --assets_dir ~/assets
```

#### For MLCommons/peoples_speech you may need to add a line to run.py to filter out the longer sequences

```
dataset = dataset.filter(lambda example: example['duration_ms'] < 30000 and example['duration_ms'] > 10000)
```

Then you can run:

```
python3 run.py --engine_dir $output_dir --dataset MLCommons/peoples_speech --dataset_name microset --enable_warmup --name peoples_speech --dataset_split train --assets_dir ~/assets  --batch_size 64

# for bigger dataset 
python3 run.py --engine_dir $output_dir --dataset MLCommons/peoples_speech --dataset_name validation --dataset_split validation --enable_warmup --name peoples_speech --assets_dir ~/assets --batch_size 64
```

## Ansible collection

From the root of the repository run:

- Install the collection:

```
cd psap/topsail
ansible-galaxy collection build --force --output-path releases/
VERSION=$(grep '^version: ' ./galaxy.yml | awk '{print $2}')
ansible-galaxy collection install releases/psap-topsail-$VERSION.tar.gz --force
```

- Run the playbook:

```
ansible-playbook playbook.yml
```
