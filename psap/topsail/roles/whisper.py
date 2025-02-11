import sys

from core.library.ansible_toolbox import (
    RunAnsibleRole, AnsibleRole,
    AnsibleMappedParams, AnsibleConstant,
    AnsibleSkipConfigGeneration
)

class Whisper:
    """
    Commands related to the current role
    """

    @AnsibleRole("whisper")
    @AnsibleMappedParams
    def main(self,
                    namespace = "my-whisper-runtime",
                    pod_name = "vllm-standalone",
                    container_name = "vllm-standalone",
                    image = "quay.io/psap/whisper-poc:latest-vllm",
                    commands_to_run = [
                        "mkdir -p /tmp/output/",
                        "nvidia-smi > /tmp/output/gpu_status.txt",
                        "python /workspace/scripts/run_vllm.py --model large-v3 > /tmp/run_vllm.log 2>&1",
                        "python /workspace/scripts/run_vllm_plot.py"
                    ],
                    results_folder_path = "/tmp/output",
                    output_folder_path = "./whisper_bench-output",
                     ):
        """
        Run the whisper role

        Args:
          namespace: the benchmark's namespace
          pod_name: the benchmark's pod name
          container_name: the benchmark's container name
          image: the benchmark's container image location
          commands_to_run: the benchmark's commands to run
          results_folder_path: the benchmark's output folder path relative to the running container
          output_folder_path: the benchmark's output folder relative to the ansible playbook
        """

        # if runtime not in ("standalone-tgis", "vllm"):
        #     raise ValueError(f"Unsupported runtime: {runtime}")

        return RunAnsibleRole(locals())
