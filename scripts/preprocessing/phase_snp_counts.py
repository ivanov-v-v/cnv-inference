import argparse
import os
import warnings

import pandas as pd

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.core.common.SettingWithCopyWarning)

import xclone.lib.pandas_utils
import xclone.lib.system_utils
from xclone.lib.workspace.workspace_manager import WorkspaceManager
import xclone.lib.preprocessing.phase_snp_counts as xclone_snp_phasing

parser = argparse.ArgumentParser(description="Takes in raw SNP count depths (AD and DP) and phases those.")
parser.add_argument("--sample", type=str, help="Sample name (must correspond to existing workspace)")
parser.add_argument("--modality", type=str, help="Modality (must correspond to existing workspace)")
parser.add_argument("--chromosomes", type=str, help="Space-separated list of chromosomes")
args = parser.parse_args()
args.chromosomes = args.chromosomes.split()

# Load the workspace

workspace = WorkspaceManager(
    task_name="preprocessing",
    experiment_info={"sample" : args.sample,
                     "modality" : args.modality},
    verbose=True
)
workspace.load_workspace()

# Filter rows by chromosome list
# and drop out non-phased SNPs afterwards

data = workspace.load_data(["raw_snp_counts", "phasing", "blocks"])
data = xclone_snp_phasing.filter_chromosomes(data, args.chromosomes)
data = xclone_snp_phasing.drop_non_phased_snps(data)
data = xclone_snp_phasing.phase_snp_counts(data)

# Load the counts into processed files dir
# and update the workspace file on disk.

xclone.lib.system_utils.pickle_dump(
    data["snp_counts"],
    os.path.join(
        workspace.tmp_dir,
        "snp_counts.pkl"
    )
)
workspace.add_entry("snp_counts", "snp_counts.pkl")
workspace.verify()
workspace.push()
workspace.write_config()