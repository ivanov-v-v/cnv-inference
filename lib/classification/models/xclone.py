from collections import namedtuple
from copy import copy, deepcopy
import os
import sys
from time import time
from typing import Dict

import numpy as np
import scipy as sp
import scipy.stats as sps
from tqdm import tqdm

import classification.models.xclone_routines as xclone_routines
import util

class XCloneDistrib:
    """
    This class stores all parameters of the XClone model.
    In a sense, this class represents the whole graphical model:
    it doesn't only store the parameters, but also initializes the 
    distributions and provides update rules.
    """
    
    def __init__(
        self, A_G_prime: np.ndarray, 
        D_G_prime: np.ndarray, 
        A_G: np.ndarray, 
        D_G: np.ndarray, 
        I_G_prime: np.array, 
        CNV_prime: np.ndarray, 
        T_max: int
    ):
        """
        :param A_G_prime: alternative allele count depth, scDNA
        :param D_G_prime: reference allele count depth, scDNA
        :param A_G: alternative allele count depth, scRNA
        :param D_G: reference allele count depth, scRNA
        :param I_G_prime: cell-to-clone assignment, scDNA
        :param CNV_prime: copy number state of blocks, scDNA
        :param T_max: maximal possible value of a CNV state 
                        (all larger values are clipped)
        """
        
        assert A_G_prime.shape == D_G_prime.shape,\
            "scDNA count matrices have incompatible shapes: "\
            f"{A_G_prime.shape} and {D_G_prime.shape} for AD and DP respectively"
        assert A_G.shape == D_G.shape,\
            "scRNA count matrices have incompatible shapes: "\
            f"{A_G.shape} and {D_G.shape} for AD and DP respectively"
        assert A_G.shape[0] == A_G_prime.shape[0],\
            "Different number of blocks in scDNA and scRNA count matrices: "\
            f"{A_G_prime.shape[0]} and {A_G.shape[0]} respectively"
        assert I_G_prime.size == A_G_prime.shape[1],\
            "Cell labels and count matrix (scDNA) have incompatible shaped: "\
            f"{I_G_prime.shape} and {A_G_prime.shape} respectively."
       
        self._init_count_depths(
            A_G_prime=A_G_prime, 
            D_G_prime=D_G_prime, 
            A_G=A_G, 
            D_G=D_G
        )
        self._init_clonal_labels(I_G_prime=I_G_prime)
        self._init_CNV(CNV_prime=CNV_prime, T_max=T_max)
        
        # ASR for each block in each CNV configuration
        # is assumed to follow the beta distribution.
        # We explicitly store (alpha, beta) parameters 
        # of Beta(alpha, beta) and update those during the runtime
        
        self._init_ASR()
    
    def _init_count_depths(self, A_G_prime, D_G_prime, A_G, D_G):
        self.N_G = A_G.shape[0]
        
        # Missing count depths are explicitly zero-imputed for now.
        # We will switch to sparse matrices eventually
        
        self.A_G_prime = np.nan_to_num(A_G_prime.astype(np.float64))
        self.D_G_prime = np.nan_to_num(D_G_prime.astype(np.float64))
        
        # I decided to introduce 'reference allele count depths',
        # because it speeds up recurring computations
        
        self.R_G_prime = self.D_G_prime - self.A_G_prime
        self.M_prime = self.A_G_prime.shape[1]
        
        # Same for scRNA
        
        self.A_G = np.nan_to_num(A_G.astype(np.float64))
        self.D_G = np.nan_to_num(D_G.astype(np.float64))
        self.R_G = self.D_G - self.A_G
        self.M = A_G.shape[1] 
        
        # We preocumpute binomial coefficients 
        # to speed-up computation of likelihood
        # in binomial model. 
        
        self.bincoeff_prime = (
            xclone_routines.logfact(self.D_G_prime) 
            - xclone_routines.logfact(self.A_G_prime) 
            - xclone_routines.logfact(self.R_G_prime)
        )

        self.bincoeff = (
            xclone_routines.logfact(self.D_G) 
            - xclone_routines.logfact(self.A_G) 
            - xclone_routines.logfact(self.R_G)
        )
       
    def _aggregate_by_clone(self, mx, labels, agg_fn):
        """
        Aggregate cell-specific information in accordance
        with the clonal label assignment. Can be used to
        compute clonal count matrices by adding up AD/DP counts.
        
        :param mx: numberical numpy matrix, N x M
        :param labels: clonal label assignment, numpy array of size M
        :param agg_fn: ufunc 
        """
        assert labels.size == mx.shape[1],\
            "Shape of passed labels doesn't match the shape of the matrix"
        clones = np.unique(labels)
        return np.column_stack(
            agg_fn(mx[:, labels == k])
            for k in clones
        )   
      
    def _init_clonal_labels(self, I_G_prime):
        """
        Initializes such parameters as:
        1. Cell-to-clone label assignments I_G_prime, I_G
        2. Number of clones K and their frequencies f
        3. Clonal count matrices A_C_prime, R_C_prime and D_C_prime
        """
        
        self.I_G_prime = I_G_prime.astype(np.int64)
        
        # We also need to define 'clonal' AD/DP profiles.
        # We need this to ensure efficient likelihood computation.
        # We proceed by adding up counts of all cells sharing the same label.
        # It's questionable that this is the best aggregation strategy,
        # but it is well aligned with out current probabilistic model.
        
        self.clones, label_counts = np.unique(self.I_G_prime, 
                                              return_counts=True)
        
        colsum_fn = lambda mx: np.sum(mx, axis=1)
        self.A_C_prime = self._aggregate_by_clone(
            self.A_G_prime, 
            self.I_G_prime, 
            colsum_fn
        )
        self.D_C_prime = self._aggregate_by_clone(
            self.D_G_prime, 
            self.I_G_prime, 
            colsum_fn
        )
        self.R_C_prime = self.D_C_prime - self.A_C_prime
        
        self.K = self.clones.size
        self.f = label_counts / self.M_prime
        assert np.isclose(self.f.sum(), 1),\
            "Clonal frequencies don't add up to 1 (wtf?)"
        
        # Initial clonal labels of cells in scRNA dataset
        # are sampled in accordance with observed label frequencies
        
        self.I_G = sps.rv_discrete(
           a=0, # lower bound
           b=self.K, # upper bound
           values=[np.arange(self.K), self.f] # probabilities
        ).rvs(size=self.M)
        
        
    def _init_CNV(self, CNV_prime, T_max):
        """
        Initalizes the following parameters:
        1. CNV_prime — raw CNV segments in cells from scDNA sample
        2. T_max — maximal total CNV number (all values above are clipped)
        3. T — clonal CNV profiles
        """
        self.CNV_prime = CNV_prime
        self.T_max = T_max
        
        # CNV number clipping: everything above T_max is 
        # considered to be a noise and is replaced with T_max
        
        self.CNV_prime[self.CNV_prime > self.T_max] = self.T_max
        
        # All the possible chromosomal configurations
        # (n_maternal, n_paternal) with a constraint that
        # n_maternal + n_paternal <= T_max
        
        self.tau = np.concatenate([[(t - k, k) for k in range(t + 1)] 
                                   for t in range(1, T_max + 1)])
        
        # Can we impute missing values with some default?
        # With normal configuration of (1, 1), for example.
        
        self.T = xclone_routines.init_T(
            A_C_prime=self.A_C_prime, 
            D_C_prime=self.D_C_prime, 
            CNV_prime=self.CNV_prime, 
            N_G=self.N_G, 
            K=self.K
        )
        
    def _init_ASR(self):
        """
        Initialize parameters of the beta prior.
        Also initialize H_* and X_* — currently we prefer to 
        precompute these mappings and store results in numpy matrices.
        This simplifies likelihood computations.
        """
        
        self.Alpha_G, self.Beta_G = xclone_routines.init_alpha_beta(
            self.N_G, 
            self.tau
        )
        self.Theta_G = sps.beta(
            a=self.Alpha_G, 
            b=self.Beta_G
        ).rvs(size=(self.Alpha_G.shape))
        
        # This looks ugly, and I am not 100% sure that we need it this way,
        # but I decided to store everything as numpy matrices...
        # The problem with this approach is that we need to recompute
        # the H_* and X_* matrices on every posterior update.
        # If H_* and X_* were functions, this problem wouldn't exist.
        
        self.H_G_prime, self.X_G_prime = xclone_routines.init_H_X(
            N=self.N_G, 
            M=self.M_prime, 
            I=self.I_G_prime,
            T=self.T, 
            Theta_G=self.Theta_G
        )
        
        self.H_C_prime, self.X_C_prime = xclone_routines.init_H_X(
            N=self.N_G, 
            M=self.K, 
            I=np.arange(self.K),
            T=self.T, 
            Theta_G=self.Theta_G
        )
        
        self.H_G, self.X_G = xclone_routines.init_H_X(
            N=self.N_G, 
            M=self.M, 
            I=self.I_G,
            T=self.T, 
            Theta_G=self.Theta_G
        )
       
        
    def total_loglikelihood(self):
        """
        Wrapper around the JIT-compiled routine that computes
        the joint loglikelihood under current clone label assignment.
        """ 
        return xclone_routines.total_loglikelihood(
            self.A_G_prime, self.R_G_prime, self.X_G_prime, self.bincoeff_prime,
            self.A_G, self.R_G, self.X_G, self.bincoeff
        )
    
    def copy(self):
        """
        Carefully creates a snapshot of a parameter space. 
        Constant parts (AD/DP matrices, I_G_prime) are shared by reference.
        Mutable ones (I_G, H_*, X_*, Alpha_G, Beta_G, Theta_G) are deep-copied.
        """
        
        # TODO: ENSURE THAT PARAMETERS WHICH ARE NOT EXCPLICITLY COPIED
        # REMAIN CONSTANT (AS IF THEY WERE PASSED BY CONST REFERENCE)
        
        params_copy = copy(self)
        
        params_copy.Alpha_G = deepcopy(self.Alpha_G)
        params_copy.Beta_G = deepcopy(self.Beta_G)
        params_copy.Theta_G = deepcopy(self.Theta_G)
        
        params_copy.I_G = deepcopy(self.I_G)
      
        params_copy.H_G_prime = deepcopy(self.H_G_prime)
        params_copy.X_G_prime = deepcopy(self.X_G_prime)
        
        params_copy.H_C_prime = deepcopy(self.H_C_prime)
        params_copy.X_C_prime = deepcopy(self.X_C_prime)
        
        params_copy.H_G = deepcopy(self.H_G)
        params_copy.X_G = deepcopy(self.X_G)
        
        return params_copy

class XCloneGibbsSampler:
    """
    This class performs posterior updates.
    It updates all parameters on each iteration.
    Sampler doesnt's store the parameters. 
    They must be passed by reference to each of the update rules.
    This increases the verbosity of function declarations
    but I don't want to mix the storage class for parameters
    of the model with the class that updates those.
    """
    
    def update_ASR(self, params, A_G, D_G, H_G, changed_mask):
        """
        Update ASR posterior. Only propagate the evidence coming
        from reassigned cells (specified by boolean mask changed_mask). 
        Recompute H_* and X_* maps. 
        """
        params.Alpha_G, params.Beta_G = xclone_routines.update_alpha_beta(
            tau=params.tau,
            Theta_G=params.Theta_G,
            Alpha_G=params.Alpha_G,
            Beta_G=params.Beta_G,
            A_G=A_G,
            D_G=D_G,
            H_G=H_G,
            changed_mask=changed_mask
        )
        
        params.Theta_G = sps.beta(
            a=params.Alpha_G, 
            b=params.Beta_G
        ).rvs(size=params.Theta_G.shape)
        
        # TODO: THIS IS VERBOSE. DO WE REALLY NEED TO PRECOMPUTE THOSE?
        # WON'T UFUNCS WORK AS WELL?
        
        params.H_G_prime, params.X_G_prime = xclone_routines.init_H_X(
            N=params.N_G, 
            M=params.M_prime, 
            I=params.I_G_prime,
            T=params.T, 
            Theta_G=params.Theta_G
        )

        params.H_C_prime, params.X_C_prime = xclone_routines.init_H_X(
            N=params.N_G, 
            M=params.K, 
            I=np.arange(params.K),
            T=params.T, 
            Theta_G=params.Theta_G
        )
    
        params.H_G, params.X_G = xclone_routines.init_H_X(
            N=params.N_G, 
            M=params.M, 
            I=params.I_G,
            T=params.T, 
            Theta_G=params.Theta_G
        )
        
    def update_I_G(self, params: XCloneDist):
        """
        Sample new clonal labels for each cell in scRNA
        in accordance with the probability distribution over the classes
        defined by the log-likelihoods of sampling each particular 
        cell from each of the ASR profiles.
        """
        prior_I_G = deepcopy(params.I_G)
        params.I_G = xclone_routines.update_I_G(
            A=params.A_G,
            R=params.R_G,
            D=params.D_G,
            X_C_prime=params.X_C_prime,
            f=params.f
        )
        changed_mask = params.I_G != prior_I_G
        
        # We need to update the posterior 
        # by propagating evidence coming from reassigned cells
        
        self.update_ASR(
            params,
            A_G=params.A_G, 
            D_G=params.D_G, 
            H_G=params.H_G,
            changed_mask=changed_mask
        )
        
    def sample(self, params: XCloneDist):
        """
        Make a snapshot of current model parameters,
        sample new clonal labels, update the posterior
        in a copy and return it back.
        """
        new_params = params.copy()
        self.update_I_G(new_params)
        return new_params
    
class XClone:
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
        self, A_G_prime: np.ndarray, 
        D_G_prime: np.ndarray, 
        A_G: np.ndarray, 
        D_G: np.ndarray, 
        I_G_prime: np.array, 
        CNV_prime: np.ndarray, 
        T_max: int,
        report_dir: str,
        verbose: bool = False
    ):
        """
        :param A_G_prime — alternative allele count depth, scDNA
        :param D_G_prime — total count depth, scDNA
        :param A_G — alternative allele count depth, scRNA
        :param D_G — total count depth, scRNA
        :param I_G_prime — cell-to-clone assignment, scDNA
        :param CNV_prime — copy number state of blocks, scDNA
        :param T_max — maximal possible value of a CNV state 
                        (all larger values are clipped)
        """
        self._params = XCloneDistrib(
            A_G_prime, D_G_prime,
            A_G,  D_G, 
            I_G_prime, 
            CNV_prime, T_max
        )
        
        self._sampler = XCloneGibbsSampler()
        
        self._sampler.update_ASR(
            params=self._params,
            A_G=self._params.A_G_prime,
            D_G=self._params.D_G_prime,
            H_G=self._params.H_G_prime,
            changed_mask=np.ones(self._params.M_prime, dtype=np.bool)
        )
        
        self._report_dir = report_dir
        self._tqdm_path = f"{self._report_dir}/tqdm.txt"
        self._log_path = f"{self._report_dir}/log.txt"
        with open(self._log_path, "w") as log_file:
            log_file.write("Negloglikelihood\t%Reassigned\tIteration")
        
        self._log_stream = sys.stdout if verbose else open(os.devnull, "w")
        
        self._best_loglik = self._params.total_loglikelihood()
        self._changed_mask = np.ones(self._params.K)
        self._iter_count = 0
    
    def dump(self):
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
        util.pickle_dump(self._params, path_to_dump)
        
    def write_log(self, log_file):
        """
        Writes down such quantities as:
        1. current best negative log likelihood;
        2. fraction of cells whose labels were reassigned;
        3. number of training iterations passed so far.
        """
        log_file.write(
            "{}\t{:.3f}\t{:.3f}\n".format(
                -self._best_loglik
                100 * np.mean(self._changed_mask),
                self._iter_count,
            )
        )
        log_file.flush()
    
    def fit(self, n_iters, callback):
        """
        Repeatedly samples clonal label assignments
        until the posterior distribution converges.
        
        :param n_iters: Number of sampling iterations. Currently, resetting 
                        XClone's inner state is not an option, so each call
                        of the `fit` function starts exactly where it left
                        off at the end of the previous call (even if interrupted)
        :param callback: Function that is triggered on successfull labelling 
                         update. It takes XClone instance does something useful,
                         but should not modify its state in any way.
        """
        self.dump()
        
        with open(self._tqdm_path, "w") as tqdm_file,\
                open(self._log_path, "w+") as log_file:
            
            for _ in tqdm(range(n_iters), file=tqdm_file):
                new_params = self._sampler.sample(self._params)
                new_loglik = new_params.total_loglikelihood()
                
                if new_loglik > self._best_loglik:
                    self._changed_mask = self._params.I_G != new_params.I_G
                    callback(self)
                    
                    print(
                        "Iteration {} — labelling update!" 
                        "{:.2f} --> {:.2f}, {:.0f}% labels reassigned".format(
                            self._iter_count,
                            self._best_loglik,
                            new_loglik,
                            100 * np.mean(self._changed_mask),
                        ),
                        file=self._log_stream
                    )
                    
                    self._best_loglik = new_loglik
                    self._params = new_params.copy()
                    self.dump()
                    self.write_log(log_file)
                
                self._iter_count += 1