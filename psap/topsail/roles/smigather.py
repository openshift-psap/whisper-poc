import sys

from core.library.ansible_toolbox import (
    RunAnsibleRole, AnsibleRole,
    AnsibleMappedParams, AnsibleConstant,
    AnsibleSkipConfigGeneration
)

class Smigather:
    """
    Commands related to the current role
    """

    @AnsibleRole("smi_gather")
    @AnsibleMappedParams
    def main(self,
                    dcgm_namespace="nvidia-gpu-operator",         # Namespace where DCGM pod is located
                    dcgm_pod_label="nvidia-dcgm-exporter",        # Label or prefix of the DCGM pod
                    container_name_suffix="nvidia-dcgm-exporter", # DCGM container name suffix
                    monitor_interval=1,                    # Time in seconds between nvidia-smi captures
                    container_output_file="/tmp/gpu_metrics.csv",
                    local_output_file="./whisper_bench-output/gpu_metrics.csv",
                    output_folder_path="./whisper_bench-output",
                    command_pod_namespace="my-whisper-runtime", # Namespace of the monitored command pod
                    command_pod_name="vllm-standalone",    # Name of the monitored command pod
                     ):
        """
        Run the Smigather role

        Args:
          dcgm_namespace: the namespace where the dcgm exporter container is running
          dcgm_pod_label: the label of the pod running the dcgm exporter
          container_name_suffix: the dcgm container suffix
          monitor_interval: the interval for retrieving data from the dcgm exported container
          container_output_file: the output file when retrieving data from the dcgm container
          local_output_file: the output file relative to the playbook execution
          output_folder_path: the folder path relative to the playbook execution
          command_pod_namespace: the namespace of the pod running the benchmark
          command_pod_name: the name of the pod running the benchmark
        """

        # if runtime not in ("standalone-tgis", "vllm"):
        #     raise ValueError(f"Unsupported runtime: {runtime}")

        return RunAnsibleRole(locals())
