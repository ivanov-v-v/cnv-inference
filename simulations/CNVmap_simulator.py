# Simulator for generating allelic count matrices for
# both single-cell DNA-seq and single-cell RNA-seq
# Author: Yuanhua Huang
# Date: 30-11-2019

import numpy as np

def CNV_simulator(tau, T_mat, DP_RNA_seed, DP_DNA_seed,
                  n_cell_DNA=200, n_cell_RNA=200, 
                  beta_shape_DNA=30, beta_shape_RNA=1,
                  share_theta=True, random_seed=None):
    """
    First version, not supporting CNV=0
    
    tau: (n_state, 2), array_like of ints
        copy number of m & p for each CNV state.
        In future, we could change this variable 
        to theta_prior
    T_mat: (n_block, n_clone), array_like of ints
        clone configuration of copy number states
    
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_state = tau.shape[0]
    n_block = T_mat.shape[0]
    n_clone = T_mat.shape[1]

    ## generate cell cluster labels; uniform distribution
    I_RNA = np.random.choice(n_clone, size=n_cell_RNA)
    I_DNA = np.random.choice(n_clone, size=n_cell_DNA)
    
    ## generate Theta parameter for ASR; beta distribution
    _base = 0.01  # avoiding theta to be 0 or 1
    _theta_prior = (tau[:, 0] + _base) / (tau + _base).sum(axis=1)
    
    Theta_DNA = np.zeros((n_block, n_state))
    Theta_RNA = np.zeros((n_block, n_state))
    for j in range(Theta_DNA.shape[1]):
        _s1 = beta_shape_DNA * _theta_prior[j]
        _s2 = beta_shape_DNA * (1 - _theta_prior[j])
        Theta_DNA[:, j] = np.random.beta(_s1, _s2, size=n_block)
        
        if share_theta:
            Theta_RNA[:, j] = Theta_DNA[:, j]
        else:
            _s1 = beta_shape_RNA * Theta_DNA[:, j]
            _s2 = beta_shape_RNA * (1 - Theta_DNA[:, j])
            Theta_RNA[:, j] = np.random.beta(_s1, _s2)
    
    ## Generate DP matrix: uniform distribution, profile reserved
    idx_DP_RNA = np.random.choice(DP_RNA_seed.shape[1], size=n_cell_RNA)
    idx_DP_DNA = np.random.choice(DP_DNA_seed.shape[1], size=n_cell_DNA)
    
    DP_RNA = DP_RNA_seed[:, idx_DP_RNA].astype(int)
    DP_DNA = DP_DNA_seed[:, idx_DP_DNA].astype(int)
    
    ## Generate X and AD matrices: binomial distribution
    X_RNA = np.zeros(DP_RNA.shape)
    for i in range(n_block):
        for j in range(n_cell_RNA):
            X_RNA[i, j] = Theta_RNA[i, int(T_mat[i, I_RNA[j]])]
    AD_RNA = np.random.binomial(DP_RNA, X_RNA)
    
    X_DNA = np.zeros(DP_DNA.shape)
    for i in range(n_block):
        for j in range(n_cell_DNA):
            X_DNA[i, j] = Theta_DNA[i, int(T_mat[i, I_DNA[j]])]
    AD_DNA = np.random.binomial(DP_DNA, X_DNA)
            
    ## return values
    RV = {}
    RV["tau"] = tau
    RV["T_mat"] = T_mat
    RV["I_RNA"] = I_RNA
    RV["I_DNA"] = I_DNA
    RV["X_RNA"] = X_RNA
    RV["X_DNA"] = X_DNA
    RV["DP_RNA"] = DP_RNA
    RV["DP_DNA"] = DP_DNA
    RV["AD_RNA"] = AD_RNA
    RV["AD_DNA"] = AD_DNA
    RV["Theta_RNA"] = Theta_RNA
    RV["Theta_DNA"] = Theta_DNA
    return RV
