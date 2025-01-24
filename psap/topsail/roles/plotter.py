import sys

from core.library.ansible_toolbox import (
    RunAnsibleRole, AnsibleRole,
    AnsibleMappedParams, AnsibleConstant,
    AnsibleSkipConfigGeneration
)

class Plotter:
    """
    Commands related to the current role
    """

    @AnsibleRole("plotter")
    @AnsibleMappedParams
    def main(self,
                     csv_file_path="whisper_bench-output/gpu_metrics.csv",
                     ):
        """
        Run the plotter role

        Args:
          csv_file_path: the path where the role will be fetching the bench output data
        """

        # if runtime not in ("standalone-tgis", "vllm"):
        #     raise ValueError(f"Unsupported runtime: {runtime}")

        return RunAnsibleRole(locals())
