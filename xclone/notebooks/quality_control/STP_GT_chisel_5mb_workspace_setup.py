import xclone_config
project_config = xclone_config

from xclone.lib.workspace.workspace_manager import WorkspaceManager
import xclone.lib.system_utils

modality_tag = "scDNA"
sample_tag = "STP_GT"
context_tag = "chisel_5mb"

workspace = WorkspaceManager(
    task_name="quality_control",
    experiment_info={
        "modality" : modality_tag,
        "sample" : sample_tag   
    },
    cookiecutter_info=xclone.lib.system_utils.load_cookiecutter_info(project_config),
    verbose=True
)
requirements = {
    "blocks" : f"{context_tag}/blocks.pkl",
    "ase" : f"{context_tag}/baf.pkl",
    "scCNV" : f"{context_tag}/copy_number_aberrations.pkl",
    "bin_counts" : f"{context_tag}/block_counts.pkl",
    "block_counts" : "block_counts.pkl",
    "clustering" : f"{context_tag}/clustering.pkl"
}
workspace.prepare_workspace(requirements)
