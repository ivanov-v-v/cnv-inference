import xclone_config
project_config = xclone_config

from xclone.lib.workspace.workspace_manager import WorkspaceManager
import xclone.lib.system_utils

modality_tag = "scDNA"
sample_tag = "CHISEL_PatientS0"

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
    "blocks" : "blocks.pkl",
    "ase" : "baf.pkl",
    "scCNV" : "copy_number_aberrations.pkl",
    "block_counts" : "block_counts.pkl",
    "clustering" : "clustering.pkl"
}
workspace.prepare_workspace(requirements)
