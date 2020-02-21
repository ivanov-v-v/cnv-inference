from collections import defaultdict
from datetime import datetime
import os
import sys
import shutil

import numpy as np
from tqdm import tqdm

import system_utils
from workspace.workspace_manager import WorkspaceManager
from classification.xclone_model.xclone_distrib import XCloneDistrib
from classification.xclone_model.xclone_gibbs_sampler import XCloneGibbsSampler
import classification.xclone_model.xclone_routines as xclone_routines

class XCloneEngine:
    """        
    This is an up-to-date implementation of XClone model.
    All the naming conventions are synchronized with the manuscript.
    Heavy computations are delegated to JIT-compiled functions in
    xclone_routines. Model initalization is delegated to the storage
    class XCloneDist. Posterior updates are handled by a separate class.
    Currently only Gibbs sampling is supported, but we plan to switch
    to variational inference (to make computations more feasible).
    """
        
    def __init__(
        self,
        workspace_dict,
        params: XCloneDistrib,
        sampler: XCloneGibbsSampler,
        criterion,
        report_dir: str,
        verbose: bool = False
    ):
        """
        :param A_DNA — alternative allele count depth, scDNA
        :param D_DNA — total count depth, scDNA
        :param A_RNA — alternative allele count depth, scRNA
        :param D_RNA — total count depth, scRNA
        :param I_DNA — cell-to-clone assignment, scDNA
        :param CNV_DNA — copy number state of blocks, scDNA
        :param T_max — maximal possible value of a CNV state 
                        (all larger values are clipped)
        """
        self._creation_datetime = datetime.now()

        self._workspace_dict = workspace_dict
        self._params = params
        self._sampler = sampler
        self._criterion = criterion

        self._init_loglik = self._params.total_loglikelihood()

#        self._sampler.update_prior_ASR(self._params) # TODO: DEPRECATE
        self._sampler.update_posterior(self._params)
        
        self._report_dir = report_dir
        shutil.rmtree(report_dir, ignore_errors=True)
        os.makedirs(report_dir)
        self._figures_dir = os.path.join(report_dir, "figures")
        os.makedirs(self._figures_dir)
        
        self._history = defaultdict(list)
        self._tqdm_path = f"{self._report_dir}/tqdm.txt"
        self._log_path = f"{self._report_dir}/log.txt"
        with open(self._log_path, "w") as log_file:
            log_file.write("Iteration\tNegloglikelihood\t%Reassigned\n")
        
        self._log_stream = sys.stdout if verbose else open(os.devnull, "w")
        
        self._best_loglik = self._params.total_loglikelihood()
        self._changed_mask = np.ones(self._params.K)
        self._iter_count = 0

    def __str__(self):
        return (
            "XCloneEngine, 2019-2020, Stegle Lab\n"
            f"Report dir: {self._report_dir}\n"
            f"This instance was created on {self._creation_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"Criterion: {self._criterion}\n"
            f"DNA sample: {self._workspace_dict['scDNA'].sample}\n"
            f"Cells in DNA sample: {self._params.M_DNA}\n"
            f"RNA sample: {self._workspace_dict['scRNA'].sample}\n"
            f"Cells in RNA sample: {self._params.M_RNA}\n"
            f"Total number of CNV-haploblocks: {self._params.N}\n"
            f"Number of clones: {self._params.K}\n"
            f"{self._iter_count} fitting iterations already passed\n"
            f"Sampler used: {type(self._sampler)}\n"
            f"Initial total loglikelihood: {self._init_loglik}\n"
            f"Current best total loglikelihood: {self._best_loglik}\n"
        )

    def __repr__(self):
        return str(self)
    
    def save(self):
        """
        Takes a snapshot of the current model parameters
        and dumps it to the file in report_dir. Dumping should
        only be triggered by successfull relabelling.
        
        Name of the dump includes:
        1. current best negative log likelihood;
        2. fraction of cells whose labels were reassigned;
        3. number of training iterations passed so far.
        """
        path_to_dump = "{}/{:.3f}_{:.3f}_{}.pkl".format(
            self._report_dir,
            -self._best_loglik,
            100 * np.mean(self._changed_mask),
            self._iter_count
        )
        system_utils.pickle_dump(self._params, path_to_dump)

    @staticmethod
    def load(path_to_pickle):
        """
        Reconstruct the model from the checkpoint.
        :param path_to_pickle: path to the pickle file
        :return: loaded XClone instance
        """
        return system_utils.pickle_load(path_to_pickle)
    
    def fit(self, n_iters, convergence_tracker):
        """
        Repeatedly samples clonal label assignments
        until the posterior distribution converges.
        
        :param n_iters: Number of sampling iterations. Currently, resetting 
                        XClone's inner state is not an option, so each call
                        of the `fit` function starts exactly where it left
                        off at the end of the previous call (even if interrupted)
        :param convergence_tracker: 
        """
        self.save()
        
        with open(self._tqdm_path, "w+") as tqdm_file,\
                open(self._log_path, "a+") as log_file:
            
            for _ in tqdm(range(n_iters), file=tqdm_file):
#                 if self._iter_count % 500 == 0 and self._iter_count > 0:
#                     most_likely_labels = np.array([
#                         np.argmax(xclone_routines.compute_assignment_probas(
#                             cell_id,
#                             A=self._params.A_RNA,
#                             R=self._params.R_RNA,
#                             X_CLONE=self._params.X_CLONE,
#                             logbincoeffs=self._params.logbincoeff_RNA,
#                             f=self._params.f
#                         )) for cell_id in range(self._params.M_RNA)
#                     ])
#                     self._candidate_params = self._params.copy()
#                     self._candidate_params.I_RNA = most_likely_labels
#                     self._sampler.update_H_RNA(self._candidate_params)
#                 else:
                self._candidate_params = self._sampler.sample(self._params)
                self._candidate_loglik = self._candidate_params.total_loglikelihood()
                convergence_tracker.make_snapshot(self)

                if self._iter_count % 100 == 0 and self._iter_count > 0:
                    convergence_tracker.perform_diagnostics(self)

                if self._criterion(self):
                    self._history["updates"].append(self._iter_count)
                    self._best_loglik = self._candidate_loglik
                    self._params = self._candidate_params.copy()
                    convergence_tracker.perform_diagnostics(self)
                    self.save()

                self._iter_count += 1
