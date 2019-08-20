from collections import defaultdict
from datetime import datetime
import os 
import shutil
import sys
from tqdm import tqdm_notebook
import json
import json2table

import cnv_inference_config
import util

class WorkspaceManager:
    def __init__(
        self,
        task_name,
        experiment_info=None,
        cookiecutter_info=None,
        verbose=False
    ):
        
        if cookiecutter_info is None:
            cookiecutter_info = util.load_cookiecutter_info(
                cnv_inference_config
            )
        
        self._nbconfig = defaultdict(dict)
        self._nbconfig["task_name"] = task_name
        self._nbconfig["experiment_info"] = experiment_info
        self._nbconfig["cookiecutter_info"] = cookiecutter_info
        self._nbconfig["config_dir"] = os.path.join(
            cookiecutter_info["notebooks"],
            task_name
        )
        self._creation_timestamp = datetime.now()
        self._logstream = sys.stderr if verbose else open(os.devnull, "w")
        self._staged_for_commit = {}
    
    @property
    def task_name(self):
        return self._nbconfig["task_name"]
       
    @property
    def experiment_info(self):
        return self._nbconfig["experiment_info"]
        
    @property
    def dir(self):
        return self._nbconfig["data_dir"]["sample"]
    
    @property
    def tmp_dir(self):
        return self._nbconfig["tmp_data_dir"]["sample"]
    
    @property
    def data(self):
        return self._nbconfig["data"]
    
    @property
    def tmp_data(self):
        return self._nbconfig["tmp_data"]
    
    @property
    def img_dir(self):
        return self._nbconfig["img_dir"]["sample"]
    
    def __str__(self):
        return ("Task name: {}\n"
                "Experiment info:\n{}\n"
                "Config generated on {}\n"
                "Contents:\n{}\n").format(
            self._nbconfig["task_name"],
            json.dumps(self._nbconfig["experiment_info"], indent=4),    
            self._creation_timestamp.strftime("%Y-%m-%d %H:%M"),
            json.dumps(self._nbconfig, indent=4)
        )
    
    def __repr__(self):
        return str(self)
    
    def _repr_html_(self):
        return "".join(
            json2table.convert(json_obj)
            for json_obj in [   
                {"Task name: " : self._nbconfig["task_name"]},
                {"Experiment info: " : self._nbconfig["experiment_info"]},    
                {"Datasets: " : list(self._nbconfig["tmp_data"].keys())},
                {"Created on " : self._creation_timestamp.strftime("%Y-%m-%d %H:%M")},
                {"Contents: " : self._nbconfig}
            ]
        )
    
    
    def prepare_workspace(self, requirements):
        self._include_data(requirements)
        self._include_tmp_data(requirements)
        self._include_img() 
        
        self._upload_tmp_data(requirements)
        self._write_config()
        
    def load_workspace(self, requirements=None):
        self._load_config()
            
        if requirements is not None:
            for data_type, filename in requirements.items():
                assert data_type in self._nbconfig["tmp_data"].keys(),\
                    f"{data_type} not loaded to workspace"

                stored_filename = os.path.basename(
                    self._nbconfig["data"].get(data_type)
                )
                assert stored_filename == filename,\
                    f"{stored_filename} was loaded"\
                    f" as {data_type} instead of {filename}"
            
    def clear_workspace(self):
        shutil.rmtree(self._nbconfig["tmp_data_dir"]["sample"])
    
    def add_entry(self, data_type, filename):
        self._staged_for_commit[data_type] = filename
        
    def remove_entry(self, data_type):
        self._staged_for_commit = {
            key : val
            for key, val in self._staged_for_commit.items()
            if key != data_type
        }
        
    def status(self):
        print("Staged for commit: ")
        for data_type, filename in self._staged_for_commit.items():
            print(f"\t-{data_type} : {filename}")
    
    def verify(self):
        for data_type in self._staged_for_commit:
            assert os.path.exists(os.path.join(
                self._nbconfig["tmp_data_dir"]["sample"],
                f"{data_type}.pkl"
            )), f"Malformed commit: {data_type} not found in workspace"
            
    def push(self):
        for data_type, filename in self._staged_for_commit.items():
            self._nbconfig["tmp_data"][data_type] = os.path.join(
                self._nbconfig["tmp_data_dir"]["sample"],
                f"{data_type}.pkl"
            )
            self._nbconfig["data"][data_type] = os.path.join(
                self._nbconfig["data_dir"]["sample"],
                filename
            )
            print("{} —> {}".format(
                self._nbconfig["tmp_data"][data_type],
                self._nbconfig["data"][data_type]
            ), file=self._logstream)
            
            shutil.copyfile(
                self._nbconfig["tmp_data"][data_type],
                self._nbconfig["data"][data_type]
            )
        
    def _include_data(self, requirements):
        print("processing read-only data directory", file=self._logstream)
        self._nbconfig["data_dir"]["root"] = \
            self._nbconfig["cookiecutter_info"]["processed"]
        self._nbconfig["data_dir"]["sample"] = os.path.join(
            self._nbconfig["data_dir"]["root"],
            self._nbconfig["experiment_info"]["sample"],
            self._nbconfig["experiment_info"]["data"]
        )
        for dirpath in self._nbconfig["data_dir"].values():
            assert os.path.exists(dirpath), f"{dirpath} doesn't exist"
            
        self._nbconfig["data"] = {
            dtype : os.path.join(
                self._nbconfig["data_dir"]["sample"],
                requirements[dtype]
            )
            for dtype in requirements.keys()
        }
        for filepath in self._nbconfig["data"].values():
            assert os.path.exists(filepath), f"{filepath} doesn't exist"
            
    def _include_tmp_data(self, requirements):
        print("processing workspace directory", file=self._logstream)
        self._nbconfig["tmp_data_dir"]["root"] = os.path.join(
            self._nbconfig["cookiecutter_info"]["tmp"], 
            self._nbconfig["task_name"]
        )
        self._nbconfig["tmp_data_dir"]["sample"] = os.path.join(
            self._nbconfig["tmp_data_dir"]["root"],
            self._nbconfig["experiment_info"]["sample"],
            self._nbconfig["experiment_info"]["data"]
        )
        self._nbconfig["tmp_data"] = {
            dtype : os.path.join(
                self._nbconfig["tmp_data_dir"]["sample"],
                f"{dtype}.pkl"
            ) for dtype in requirements.keys()
        }
        
    def _include_img(self):
        print("processing image directory", file=self._logstream)
        self._nbconfig["img_dir"]["root"] = os.path.join(
            self._nbconfig["cookiecutter_info"]["img"], 
            self._nbconfig["task_name"]
        )
        self._nbconfig["img_dir"]["sample"] = os.path.join(
            self._nbconfig["img_dir"]["root"],
            self._nbconfig["experiment_info"]["sample"]
        )
        for dirpath in self._nbconfig["img_dir"].values():
            # mkdir -p dirpath
            os.makedirs(dirpath, exist_ok=True)
    
    def _upload_tmp_data(self, requirements):
        print("loading data to workspace directory", file=self._logstream)
        shutil.rmtree(self._nbconfig["tmp_data_dir"]["sample"], 
                      ignore_errors=True)
        os.makedirs(self._nbconfig["tmp_data_dir"]["sample"]) 
        
        for dtype in tqdm_notebook(
            requirements.keys(),
            "copying files"
        ):
            print("{} —> {}".format(
                self._nbconfig["data"][dtype],
                self._nbconfig["tmp_data"][dtype]
            ), file=self._logstream)
            shutil.copyfile(
                self._nbconfig["data"][dtype],
                self._nbconfig["tmp_data"][dtype]
            )
    
    def _write_config(self):
        outfile = "{}/{}_workspace.pkl".format(
            self._nbconfig['config_dir'],
            self._nbconfig['experiment_info']["data"]
        )
        util.pickle_dump(self._nbconfig, outfile)
            
    def _load_config(self):
        infile = "{}/{}_workspace.pkl".format(
            self._nbconfig['config_dir'],
            self._nbconfig['experiment_info']["data"]
        )
        self._nbconfig = util.pickle_load(infile)