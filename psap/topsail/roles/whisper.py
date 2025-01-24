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
                     namespace,
                     delete_others=True,
                     ):
        """
        Run the plotter role

        Args:
          namespace: the namespace in which the model should be deployed
          delete_others: if True, deletes the other serving runtime/inference services of the namespace
        """

        # if runtime not in ("standalone-tgis", "vllm"):
        #     raise ValueError(f"Unsupported runtime: {runtime}")

        return RunAnsibleRole(locals())
