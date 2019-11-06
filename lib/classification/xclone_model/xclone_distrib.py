from copy import copy, deepcopy

import numpy as np
import pandas as pd
import scipy.stats as sps

import classification.xclone_model.xclone_routines as xclone_routines

class XCloneDistrib:
    """
    This class stores all parameters of the XClone model.
    In a sense, this class represents the whole graphical model:
    it doesn't only store the parameters, but also initializes the
    distributions and provides update rules.
    """

    @classmethod
    def from_dict(cls, datadict):
        instance = cls()
        instance.__dict__.update(datadict)
        return instance

    def __init__(self, datadict=None):
        """
        :param A_DNA: alternative allele count depth, scDNA
        :param D_DNA: reference allele count depth, scDNA
        :param A_RNA: alternative allele count depth, scRNA
        :param D_RNA: reference allele count depth, scRNA
        :param I_DNA: cell-to-clone assignment, scDNA
        :param CNV_DNA: copy number state of blocks, scDNA
        :param T_max: maximal possible value of a CNV state
                        (all larger values are clipped)
        """
        # assert (datadict is None) or (simulation_kwargs is None), \
        #     "Can't provide both datadict and simulation_kwargs at the same time"
        if datadict is not None:
            # assert simulation_kwargs is not None, \
            #     "No datadict and no simulation_kwargs provided at the same time"
            # self.__dict__.update(xclone_routines.simulate_G_T(**simulation_kwargs))
            assert datadict["A_DNA"].shape == datadict["D_DNA"].shape, \
                "scDNA count matrices have incompatible shapes: " \
                f"{datadict['A_DNA'].shape} and {datadict['D_DNA'].shape} for AD and DP respectively"
            assert datadict['A_RNA'].shape == datadict['D_RNA'].shape, \
                "scRNA count matrices have incompatible shapes: " \
                f"{datadict['A_RNA'].shape} and {datadict['D_RNA'].shape} for AD and DP respectively"
            assert datadict["A_RNA"].shape[0] == datadict["A_DNA"].shape[0], \
                "Different number of blocks in scDNA and scRNA count matrices: " \
                f"{datadict['A_DNA'].shape[0]} and {datadict['A_RNA'].shape[0]} respectively"
            assert datadict["I_DNA"].size == datadict["A_DNA"].shape[1], \
                "Cell labels and count matrix (scDNA) have incompatible shaped: " \
                f"{datadict['I_DNA'].shape} and {datadict['A_DNA'].shape} respectively."
            assert datadict["CNV_DNA"].shape[0] == datadict["A_DNA"].shape[0], \
                "Number of blocks with known total CNV number is " \
                "different from the number of blocks with known CNV information " \
                f"({datadict['A_DNA'].shape[0]} and {datadict['CNV_DNA'].shape[0]}) respectively"

            self._init_count_depths(
                A_DNA=datadict["A_DNA"],
                D_DNA=datadict["D_DNA"],
                A_RNA=datadict["A_RNA"],
                D_RNA=datadict["D_RNA"]
            )
            self._init_clonal_labels(I_DNA=datadict["I_DNA"])
            self._init_CNV(CNV_DNA=datadict["CNV_DNA"], T_max=datadict["T_max"])

            # ASR for each block in each CNV configuration
            # is assumed to follow the beta distribution.
            # We explicitly store (alpha, beta) parameters
            # of Beta(alpha, beta) and update those during the runtime

            self._init_ASR()

    def _init_count_depths(self, A_DNA, D_DNA, A_RNA, D_RNA):
        self.N = A_RNA.shape[0]

        # Missing count depths are explicitly zero-imputed for now.
        # We will switch to sparse matrices eventually

        self.A_DNA = np.nan_to_num(A_DNA.astype(np.float64))
        self.D_DNA = np.nan_to_num(D_DNA.astype(np.float64))

        # I decided to introduce 'reference allele count depths',
        # because it speeds up recurring computations

        self.R_DNA = self.D_DNA - self.A_DNA
        self.M_DNA = self.A_DNA.shape[1]

        # Same for scRNA

        self.A_RNA = np.nan_to_num(A_RNA.astype(np.float64))
        self.D_RNA = np.nan_to_num(D_RNA.astype(np.float64))
        self.R_RNA = self.D_RNA - self.A_RNA
        self.M_RNA = A_RNA.shape[1]

        # We preocumpute binomial coefficients
        # to speed-up computation of likelihood
        # in binomial model.

        self.logbincoeff_DNA = (
                xclone_routines.logfact(self.D_DNA)
                - xclone_routines.logfact(self.A_DNA)
                - xclone_routines.logfact(self.R_DNA)
        )

        self.logbincoeff_RNA = (
                xclone_routines.logfact(self.D_RNA)
                - xclone_routines.logfact(self.A_RNA)
                - xclone_routines.logfact(self.R_RNA)
        )

    def _aggregate_by_clone(self, mx, labels, agg_fn):
        """
        Aggregate cell-specific information in accordance
        with the clonal label assignment. Can be used to
        compute clonal count matrices by adding up AD/DP counts.

        :param mx: numerical numpy matrix, N x M
        :param labels: clonal label assignment, numpy array of size M
        :param agg_fn: ufunc
        """
        assert labels.size == mx.shape[1], \
            "Shape of passed labels doesn't match the shape of the matrix"
        clones = np.unique(labels)
        return np.column_stack(
            agg_fn(mx[:, labels == k])
            for k in clones
        )

    def _init_clonal_labels(self, I_DNA):
        """
        Initializes such parameters as:
        1. Cell-to-clone label assignments I_DNA, I_RNA
        2. Number of clones K and their frequencies f
        3. Clonal count matrices A_CLONE, R_CLONE and D_CLONE
        """

        self.I_DNA = I_DNA.astype(np.int64)

        # We also need to define 'clonal' AD/DP profiles.
        # We need this to ensure efficient likelihood computation.
        # We proceed by adding up counts of all cells sharing the same label.
        # It's questionable that this is the best aggregation strategy,
        # but it is well aligned with out current probabilistic model.

        self.clones, label_counts = np.unique(self.I_DNA, return_counts=True)

        colsum_fn = lambda mx: np.sum(mx, axis=1)
        self.A_CLONE = self._aggregate_by_clone(
            self.A_DNA,
            self.I_DNA,
            colsum_fn
        )
        self.D_CLONE = self._aggregate_by_clone(
            self.D_DNA,
            self.I_DNA,
            colsum_fn
        )
        self.R_CLONE = self.D_CLONE - self.A_CLONE

        self.K = self.clones.size
        self.f = label_counts / self.M_DNA
        assert np.isclose(self.f.sum(), 1), \
            "Clonal frequencies don't add up to 1 (wtf?)"

        # Initial clonal labels of cells in scRNA dataset
        # are sampled in accordance with observed label frequencies

        self.I_RNA = sps.rv_discrete(
            a=0,  # lower bound
            b=self.K,  # upper bound
            values=[np.arange(self.K), self.f]  # probabilities
        ).rvs(size=self.M_RNA)

    def _init_CNV(self, CNV_DNA, T_max):
        """
        Initalizes the following parameters:
        1. CNV_DNA — raw CNV segments in cells from scDNA sample
        2. T_max — maximal total CNV number (all values above are clipped)
        3. T — clonal CNV profiles
        """

        assert CNV_DNA.shape == (self.N, self.K), \
            f"CNV_DNA has wrong shape: {CNV_DNA.shape}" \
            f" instead of {(self.N, self.K)}"

        self.CNV_DNA = CNV_DNA
        self.T_max = T_max

        # CNV number clipping: everything above T_max is
        # considered to be a noise and is replaced with T_max

        overflow_mask = self.CNV_DNA > self.T_max
        if np.any(overflow_mask):  # "Trying to take argmin of an empty sequence"
            self.CNV_DNA[overflow_mask] = self.T_max

        # We do not trust double-chromosome deletions

        self.CNV_DNA[self.CNV_DNA == 0] = np.nan

        # All the possible chromosomal configurations
        # (n_maternal, n_paternal) with a constraint that
        # n_maternal + n_paternal <= T_max

        self.tau = np.concatenate([[(t - k, k) for k in range(t + 1)]
                                   for t in range(1, T_max + 1)])

        # Can we impute missing values with some default?
        # With normal configuration of (1, 1), for example.

        self.T = xclone_routines.init_T(
            A_CLONE=self.A_CLONE,
            D_CLONE=self.D_CLONE,
            CNV_DNA=self.CNV_DNA,
            N=self.N,
            K=self.K
        )

        assert np.all(
            np.isnan(self.T)
            | ((self.T >= 0)
               & (self.T < self.tau.size)
               )
        ), "T matrix contains incorrect values"

    def _init_ASR(self):
        """
        Initialize parameters of the beta prior.
        Also initialize H_* and X_* — currently we prefer to
        precompute these mappings and store results in numpy matrices.
        This simplifies likelihood computations.
        """

        # Putting non-informative prior

        #         self.Alpha = np.ones(shape=(self.N, self.tau.size))
        #         self.Beta = np.ones(shape=(self.N, self.tau.size))
        self.Alpha, self.Beta = xclone_routines.init_alpha_beta(
            self.N,
            self.tau
        )
        self.Theta = sps.beta(
            a=self.Alpha,
            b=self.Beta
        ).rvs(size=(self.Alpha.shape))

        self.post_Alpha = self.Alpha.copy()
        self.post_Beta = self.Beta.copy()
        self.post_Theta = self.Theta.copy()

        # This looks ugly, and I am not 100% sure that we need it this way,
        # but I decided to store everything as numpy matrices...
        # The problem with this approach is that we need to recompute
        # the H_* and X_* matrices on every posterior update.
        # If H_* and X_* were functions, this problem wouldn't exist.

        self.H_DNA, self.X_DNA = xclone_routines.init_H_X(
            N=self.N,
            M=self.M_DNA,
            I=self.I_DNA,
            T=self.T,
            Theta=self.post_Theta
        )

        self.H_CLONE, self.X_CLONE = xclone_routines.init_H_X(
            N=self.N,
            M=self.K,
            I=np.arange(self.K),
            T=self.T,
            Theta=self.post_Theta
        )

        self.H_RNA, self.X_RNA = xclone_routines.init_H_X(
            N=self.N,
            M=self.M_DNA,
            I=self.I_RNA,
            T=self.T,
            Theta=self.post_Theta
        )

    def total_loglikelihood(self, return_addends=False):
        """
        Wrapper around the JIT-compiled routine that computes
        the joint loglikelihood under current clone label assignment.
        """
        return xclone_routines.total_loglikelihood(
            A_DNA=self.A_DNA,
            R_DNA=self.R_DNA,
            X_DNA=self.X_DNA,
            logbincoeffs_DNA=self.logbincoeff_DNA,
            A_RNA=self.A_RNA,
            R_RNA=self.R_RNA,
            X_RNA=self.X_RNA,
            logbincoeffs_RNA=self.logbincoeff_RNA,
            return_addends=return_addends
        )

    def copy(self):
        """
        Carefully creates a snapshot of a parameter space.
        Constant parts (AD/DP matrices, I_DNA) are shared by reference.
        Mutable ones (I_RNA, H_*, X_*, Alpha, Beta, Theta) are deep-copied.
        """

        # TODO: ENSURE THAT PARAMETERS WHICH ARE NOT EXCPLICITLY COPIED
        # REMAIN CONSTANT (AS IF THEY WERE PASSED BY CONST REFERENCE)

        params_copy = copy(self)

        params_copy.Alpha = deepcopy(self.Alpha)
        params_copy.Beta = deepcopy(self.Beta)
        params_copy.Theta = deepcopy(self.Theta)

        params_copy.post_Alpha = deepcopy(self.post_Alpha)
        params_copy.post_Beta = deepcopy(self.post_Beta)
        params_copy.post_Theta = deepcopy(self.post_Theta)

        params_copy.I_RNA = deepcopy(self.I_RNA)

        params_copy.H_DNA = deepcopy(self.H_DNA)
        params_copy.X_DNA = deepcopy(self.X_DNA)

        params_copy.H_CLONE = deepcopy(self.H_CLONE)
        params_copy.X_CLONE = deepcopy(self.X_CLONE)

        params_copy.H_RNA = deepcopy(self.H_RNA)
        params_copy.X_RNA = deepcopy(self.X_RNA)

        return params_copy

    @property
    def shapes(self):
        items = []
        for key, value in self.__dict__.items():
            items.append([key, np.atleast_1d(value).shape])
        return pd.DataFrame(np.vstack(items), columns=["PARAM", "SHAPE"])