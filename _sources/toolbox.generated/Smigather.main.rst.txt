:orphan:

..
    _Auto-generated file, do not edit manually ...
    _Toolbox generate command: repo generate_toolbox_rst_documentation
    _ Source component: Smigather.main


smigather main
==============

Run the Smigather role




Parameters
----------


``dcgm_namespace``  

* The namespace where the dcgm exporter container is running

* default value: ``nvidia-gpu-operator``


``dcgm_pod_label``  

* The label of the pod running the dcgm exporter

* default value: ``nvidia-dcgm-exporter``


``container_name_suffix``  

* The dcgm container suffix

* default value: ``nvidia-dcgm-exporter``


``monitor_interval``  

* The interval for retrieving data from the dcgm exported container

* default value: ``1``


``container_output_file``  

* The output file when retrieving data from the dcgm container

* default value: ``/tmp/gpu_metrics.csv``


``local_output_file``  

* The output file relative to the playbook execution

* default value: ``./whisper_bench-output/gpu_metrics.csv``


``output_folder_path``  

* The folder path relative to the playbook execution

* default value: ``./whisper_bench-output``


``command_pod_namespace``  

* The namespace of the pod running the benchmark

* default value: ``my-whisper-runtime``


``command_pod_name``  

* The name of the pod running the benchmark

* default value: ``vllm-standalone``

