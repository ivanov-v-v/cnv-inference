import numpy as np
import scipy.stats as sps

import classification.xclone_model.xclone_routines as xclone_routines
from classification.xclone_model.xclone_distrib import XCloneDistrib

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

    def update_H_RNA(self, params):
        params.H_RNA, params.X_RNA = xclone_routines.init_H_X(
            N=params.N,
            M=params.M_RNA,
            I=params.I_RNA,
            T=params.T,
            Theta=params.post_Theta
        )

    def update_H_DNA(self, params):
        params.H_DNA, params.X_DNA = xclone_routines.init_H_X(
            N=params.N,
            M=params.M_DNA,
            I=params.I_DNA,
            T=params.T,
            Theta=params.post_Theta
        )

    def update_H_CLONE(self, params):
        params.H_CLONE, params.X_CLONE = xclone_routines.init_H_X(
            N=params.N,
            M=params.K,
            I=np.arange(params.K),
            T=params.T,
            Theta=params.post_Theta
        )

    def update_Alpha_Beta(self, params, A, R, H):
        params.post_Alpha, params.post_Beta = xclone_routines.update_alpha_beta(
            tau=params.tau,
            Alpha=params.Alpha,
            Beta=params.Beta,
            A=A, R=R, H=H
        )

    def update_Theta(self, params):
        params.post_Theta = sps.beta(
            a=params.post_Alpha,
            b=params.post_Beta,
        ).rvs(size=params.Theta.shape)

    def update_prior_ASR(self, params):
        self.update_Alpha_Beta(
            params,
            A=params.A_DNA,
            R=params.R_DNA,
            H=params.H_DNA
        )
        self.update_Theta(params)

        params.Alpha = params.post_Alpha.copy()
        params.Beta = params.post_Beta.copy()
        params.Theta = params.post_Theta.copy()

        self.update_H_DNA(params)
        self.update_H_CLONE(params)
        self.update_H_RNA(params)

    def update_posterior_ASR(self, params):
        """
        Update ASR posterior. Only propagate the evidence coming
        from reassigned cells (specified by boolean mask changed_mask).
        Recompute H_* and X_* maps.
        """
        # propagate evidence from scDNA
        self.update_Alpha_Beta(
            params,
            A=params.A_RNA,
            R=params.R_RNA,
            H=params.H_RNA
        )
        self.update_Theta(params)

        self.update_H_DNA(params)
        self.update_H_CLONE(params)
        self.update_H_RNA(params)

    def update_I_RNA(self, params: XCloneDistrib):
        """
        Sample new clonal labels for each cell in scRNA
        in accordance with the probability distribution over the classes
        defined by the log-likelihoods of sampling each particular
        cell from each of the ASR profiles.
        """
        params.I_RNA = xclone_routines.update_I(
            A=params.A_RNA,
            R=params.R_RNA,
            logbincoeffs=params.logbincoeff_RNA,
            X_CLONE=params.X_CLONE,
            f=params.f
        )

        self.update_H_RNA(params)

    def sample(self, params: XCloneDistrib):
        """
        Make a snapshot of current model parameters,
        sample new clonal labels, update the posterior
        in a copy and return it back.
        """
        new_params = params.copy()
        self.update_I_RNA(new_params)
        self.update_posterior_ASR(new_params)
        return new_params
