:orphan:

..
    _Auto-generated file, do not edit manually ...
    _Toolbox generate command: repo generate_toolbox_rst_documentation
    _ Source component: Whisper.main


whisper main
============

Run the whisper role




Parameters
----------


``namespace``  

* The benchmark's namespace

* default value: ``my-whisper-runtime``


``pod_name``  

* The benchmark's pod name

* default value: ``vllm-standalone``


``container_name``  

* The benchmark's container name

* default value: ``vllm-standalone``


``image``  

* The benchmark's container image location

* default value: ``quay.io/psap/whisper-poc:latest-vllm``


``commands_to_run``  

* The benchmark's commands to run

* default value: ``['mkdir -p /tmp/output/', 'nvidia-smi > /tmp/output/gpu_status.txt', 'python /workspace/scripts/run_vllm.py --model large-v3 > /tmp/run_vllm.log 2>&1', 'python /workspace/scripts/run_vllm_plot.py']``


``results_folder_path``  

* The benchmark's output folder path relative to the running container

* default value: ``/tmp/output``


``output_folder_path``  

* The benchmark's output folder relative to the ansible playbook

* default value: ``./whisper_bench-output``

