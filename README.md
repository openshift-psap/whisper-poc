
This repository aims to guide the deployment and
testing of multimodal whisper models on both
vLLM and TensorRT, [docs](https://openshift-psap.github.io/whisper-poc/).

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
cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper
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
```

```
ansible-galaxy collection build --force --output-path releases/
VERSION=$(grep '^version: ' ./galaxy.yml | awk '{print $2}')
ansible-galaxy collection install releases/psap-topsail-$VERSION.tar.gz --force
```

- Run the playbook:

```
ansible-playbook playbook_whisper.yml
```

#### Publishing a new Topsail release

```
MY_GALAXY_API_KEY="this_is_a_very_secure_api_key_lol"
ansible-galaxy collection publish \
    releases/psap-topsail-$VERSION.tar.gz \
    --server https://galaxy.ansible.com \
    --ignore-certs \
    --verbose \
    --api-key $MY_GALAXY_API_KEY
```

## Equivalent executions

Example of how both CLIs should be aligned to run a specific role:

```
# Running from the Ansible CLI
ansible-playbook playbook_plotter.yml

# Running from the toolbox CLI
./run_toolbox.py plotter main
```


Example of how to run a playbook (End to End test scenario):

```
# Running from the Ansible CLI
ansible-playbook playbook_whisper.yml

# Running from the toolbox CLI
./run_toolbox.py tests whisper

```

### Extending the collection default variables

```
# Create a file with the extra vars
VARS_FILE="./vars.yml"

# Use small or large-v3
# TODO: the variables passed to the CLI should be fetched from the env vars if configured

cat <<EOF > $VARS_FILE
whisper_image: quay.io/psap/whisper-poc:latest-trt
whisper_commands_to_run:
  - mkdir -p /tmp/output/
  - nvidia-smi > /tmp/output/gpu_status.txt
  - bash /home/trt/scripts/trt-build-whisper.sh -m small > /tmp/trt-build-whisper.log 2>&1
  - python3 /home/trt/scripts/run_trt.py --engine_dir ~/tensorrtllm_backend/tensorrt_llm/examples/whisper/trt_engines/large_v3_max_batch_64 --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3 --assets_dir ~/assets --num_beams 1 > /tmp/run_trt.log 2>&1
  - python3 /home/trt/scripts/run_vllm_plot.py
EOF

# Running from the Ansible CLI
ansible-playbook playbook_whisper.yml -e @$VARS_FILE
```

## Logging

```

# Update the ansible configuration (ansible.cfg) accordingly.
# python -m ara.setup.ansible
# [defaults]
# callback_plugins=/usr/local/lib/python3.10/dist-packages/ara/plugins/callback
# action_plugins=/usr/local/lib/python3.10/dist-packages/ara/plugins/action
# lookup_plugins=/usr/local/lib/python3.10/dist-packages/ara/plugins/lookup

# Let's make sure the local DB is clean
ara-manage prune --confirm
ansible-playbook playbook_plotter.yml
ara-manage generate ./ara-output
```

## Documentation

[Documentation site](https://openshift-psap.github.io/whisper-poc/)

## Running this PoC for the first time?

The following steps will allow you to test this PoC.

### Clone the repository

```
git clone https://github.com/openshift-psap/whisper-poc
```

### Install the python dependencies

```
cd whisper-poc/psap/topsail
python3 -m pip install -r requirements.txt
```

### Install the collection dependencies

```
ansible-galaxy install -r requirements.yml
```

### Install TOPSAIL as an Ansible collection

```
ansible-galaxy collection build --force --output-path releases/
VERSION=$(grep '^version: ' ./galaxy.yml | awk '{print $2}')
ansible-galaxy collection install releases/psap-topsail-$VERSION.tar.gz --force
```

### Export your kubeconfig file

```
export KUBECONFIG=<the path to my kubeconfig>
```

### Run the playbook

This wil run the whisper PoC on with less resources in a Nvidia T4

```
# Create a file with the extra vars
VARS_FILE="./vars.yml"

# Use small or large-v3
# TODO: the variables passed to the CLI should be fetched from the env vars if configured

cat <<EOF > $VARS_FILE
whisper_image: quay.io/psap/whisper-poc:latest-vllm
whisper_commands_to_run:
  - mkdir -p /tmp/output/
  - nvidia-smi > /tmp/output/gpu_status.txt
  - python /workspace/scripts/run_vllm.py --model small --range 100 > /tmp/output/whisper.txt
  - python3 /home/trt/scripts/run_vllm_plot.py
EOF

# Running from the Ansible CLI
ansible-playbook playbook_whisper.yml -e @$VARS_FILE
```
