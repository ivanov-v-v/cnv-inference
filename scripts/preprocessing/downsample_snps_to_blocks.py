import argparse
import os
import warnings

import pandas as pd

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.core.common.SettingWithCopyWarning)

import xclone.lib.pandas_utils
import xclone.lib.system_utils
from xclone.lib.workspace.workspace_manager import WorkspaceManager
import xclone.lib.preprocessing.downsample_snps_to_blocks as xclone_snp_downsampling

parser = argparse.ArgumentParser(description="Takes in phased SNP count depths (AD and DP) and .")
parser.add_argument("--sample", type=str, help="Sample name (must correspond to existing workspace)")
parser.add_argument("--modality", type=str, help="Modality (must correspond to existing workspace)")
parser.add_argument("--n_jobs", type=str, help="Max number of parallel jobs to use")
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

data = workspace.load_data(["snp_counts", "phasing", "blocks"])
data = xclone_snp_downsampling.extract_snps(data)
data = xclone_snp_downsampling.intersect_snps_with_blocks(data, n_jobs=args.n_jobs)
data = xclone_snp_downsampling.compute_block_counts(data, n_jobs=args.n_jobs)

# Load the counts into processed files dir
# and update the workspace file on disk.

xclone.lib.system_utils.pickle_dump(
    data["snp_to_blocks"],
    os.path.join(workspace.tmp_dir,"snp_to_blocks.pkl")
)
workspace.add_entry("snp_to_blocks", "snp_to_blocks.pkl")

xclone.lib.system_utils.pickle_dump(
    data["block_to_snps"],
    os.path.join(workspace.tmp_dir,"block_to_snps.pkl")
)
workspace.add_entry("block_to_snps", "block_to_snps.pkl")

xclone.lib.system_utils.pickle_dump(
    data["block_counts"],
    os.path.join(workspace.tmp_dir,"block_counts.pkl")
)
workspace.add_entry("snp_counts", "block_counts.pkl")

workspace.verify()
workspace.push()
workspace.write_config()