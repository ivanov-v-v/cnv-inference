# As our algorithms are necessarily iterative at heart,
# I've had to use Python for-loops quite extensively.
# Nevertheless, I used @numba.jit decorator whenever possible
# to ensure scalability of the current implementation.
# Nevertheless, this is a questionable design that may be changed
# for better during subsequent development phases when we figure
# out how to implement the model in a Pythonic, idiomaic way.

import numba
import numpy as np
import scipy as sp
import scipy.stats as sps

@numba.jit(nopython=True)
def logfact(n):
    """
    Ramanujan's approximation of log n!
    https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
    """
    return (
        np.multiply(n, np.log(n))
        - n 
        + np.log(np.multiply(n, (1 + 4 * np.multiply(n, (1 + 2 * n))))) / 6
        + np.log(np.pi) / 2
    )

@numba.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Numba doesn't support 'p' argument of numpy.random.choice,
    so I have to use this workaround to produce JIT-compilable code.
    
    https://github.com/numba/numba/issues/2539
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    
    # TODO: PROVE CORRECTNESS
    
    return arr[np.searchsorted(
        np.cumsum(prob), 
        np.random.random(),
        side="right"
    )]

@numba.jit(nopython=True)
def cell_logits(A, R, X, logbincoeffs):
    """
    Helper function that computes the log-likelihoods
    of observing actual AD/DP count depths in binomial
    model specified by clonal ASR profile X.
    
    This function effectively computes `sps.binom(n=D, p=X).logpmf(A)`
    by using the closed-form likelihood expression.
    
    :param cell_id: id of the cell of interest, int
    :param A: alternative allele count vector, np.array
    :param R: reference allele count vector, np.array
    :param D: A + R, np.array
    :param X: clonal ASR profile, np.array
    
    """
    return logbincoeffs + np.multiply(A, np.log(X)) + np.multiply(R, np.log(1 - X))


@numba.jit(nopython=True)
def cell_loglikelihood(A, R, X, logbincoeffs):
    """
    Log-likelihood of sampling cell 'cell_id' from profile 'X'. 
    
    :param cell_id: id of the cell of interest, int
    :param A: alternative allele count vector, np.array
    :param R: reference allele count vector, np.array
    :param D: A + R, np.array
    :param X: clonal ASR profile, np.array
    
    """
    logits = cell_logits(A, R, X, logbincoeffs)
    return np.sum(logits[np.isfinite(logits)])


# TODO: write tests that compare the output with the naive implementation
# @numba.jit(nopython=True)
def total_loglikelihood(A_DNA, R_DNA, X_DNA, logbincoeffs_DNA,
                        A_RNA, R_RNA, X_RNA, logbincoeffs_RNA, return_addends=False):
    """
    Log-likelihood of observing cells from both modalities
    under current clonal label assignment. 
    
    All the input matrices are expected to be np.ndarrays.
    
    :param A_G_DNA: alternative allele count vector, scDNA, N x M_DNA
    :param R_G_DNA: reference allele count vector, scDNA, N x M_DNA
    :param X_G_DNA: clonal ASR profiles, scDNA, N x M_DNA
    :param bincoeff_DNA: logarithms of coefficients in binomial model, scDNA, N x M_DNA
    
    :param A_G: alternative allele count vector, scRNA, N x M
    :param R_G: reference allele count vector, scRNA, N x M
    :param X_G: clonal ASR profiles, scRNA, N x M
    :param bincoeff: logarithms of coefficients in binomial model, scRNA, N x M
    """
    alt_DNA = np.multiply(A_DNA, np.log(X_DNA))
    ref_DNA = np.multiply(R_DNA, np.log(1 - X_DNA))
    
    alt_RNA = np.multiply(A_RNA, np.log(X_RNA))
    ref_RNA = np.multiply(R_RNA, np.log(1 - X_RNA))
    
    logits_DNA = np.ravel(alt_DNA + ref_DNA + logbincoeffs_DNA)
    logits_RNA = np.ravel(alt_RNA + ref_RNA + logbincoeffs_RNA)

    loglik_DNA = np.sum(logits_DNA[np.isfinite(logits_DNA)])
    loglik_RNA = np.sum(logits_RNA[np.isfinite(logits_RNA)])

    if return_addends:
        return np.array([loglik_DNA, loglik_RNA])
    return loglik_DNA + loglik_RNA


def total_loglikelihood_scipy(A_DNA, D_DNA, X_DNA, A_RNA, D_RNA, X_RNA):
    """
    Same as total_likelihood but in pure scipy (and thereby not JIT-compiled).
    This function can be used as a sanity check. 
    
    TODO: Actually, I need to WRITE TESTS that ensure that results 
    of these two functions agree.
    """

    logits_DNA = sps.binom(
        n=D_DNA,
        p=X_DNA
    ).logpmf(A_DNA)
    logits_RNA = sps.binom(
        n=D_RNA,
        p=X_RNA
    ).logpmf(A_RNA)
    return np.sum(logits_DNA[np.isfinite(logits_DNA)]) \
            + np.sum(logits_RNA[np.isfinite(logits_RNA)])
#     M_DNA, M_RNA = A_DNA.shape[1], A_RNA.shape[1]
#     return (
#         np.sum([
#             cell_loglikelihood(cell_id, A_G_DNA[:, cell_id], D_G_DNA[:, cell_id], X_G_DNA[:, cell_id])
#             for cell_id in range(M_DNA)
#         ]) + np.sum([
#             cell_loglikelihood(cell_id, A_G[:, cell_id], D_G[:, cell_id], X_G[:, cell_id])
#             for cell_id in range(M)
#         ])   
#     )

@numba.jit(nopython=True)
def config_to_id(n_total, n_paternal):
    """
    Gives the index of (maternal, total - maternal) configuration
    in tau = {((1, 0), (0, 1)), ((2, 0), (1, 1), (0, 2)), ...}
    Left element of the tuple stands for the number of maternal copies,
    right element — for the number of paternal copies
    """
    return (n_total * (n_total + 1) // 2 - 1) + n_paternal


@numba.jit(nopython=True)
def cnv_config_from_asr(total_cnv, ase_ratio):
    # np.arange(t + 1) / t = (0 / t, 1 / t, ..., t / t)
    # ((0, 1), (1, 0)), ..., ((0, t), ..., (t, 0))
    # CNV number t generates exactly t different configurations
    # degenerate configuration (0, 0) is excluded, hence -1
    n_maternal = np.argmin(np.abs(
        np.arange(total_cnv + 1)[::-1] / total_cnv
        - ase_ratio
    ))
    n_paternal = total_cnv - n_maternal
    return n_maternal, n_paternal


@numba.jit(nopython=True)
def init_T(A_CLONE, D_CLONE, CNV_DNA, N, K):
    """
    Initializes T — clone-specific CNV profiles.
    The problem is, we only have total CNV numbers, not 
    (#maternal, #paternal) tuples (configuration, (m, p) below). 
    
    We have to estimate the configuration for each block in each clone.
    We do so by choosing the (m, p) tuple such that m / (m + p) is 
    as close as possible to the actual observed ASR: 
    
    (m, p) := \argmin_{x, y} | AD[block, clone] / DP[block, clone] - x/(x+y) |
    
    :param A_CLONE: allele-specific count matrix, scDNA, N x K
    :param D_CLONE: total count matrix, scDNA, N x K
    :param CNV_DNA: raw CNV profiles for individual cells, scDNA
    :param N: number of haplotype blocks
    :param K: numbe of clones
    """
    
    T = np.full((N, K), np.nan, dtype=np.float64)
    for clone_id in range(K):
        for block_id in range(N):
            t = CNV_DNA[block_id, clone_id]
            ad = A_CLONE[block_id, clone_id]
            dp = D_CLONE[block_id, clone_id]
            if ((dp == 0) or np.isnan(dp)) \
                    or ((t == 0) or np.isnan(t)):
                continue
            n_maternal, n_paternal = cnv_config_from_asr(total_cnv=t, ase_ratio=ad / dp)
            T[block_id, clone_id] = config_to_id(t, n_paternal)
    return T


@numba.jit(nopython=True)
def beta_mode(alpha, beta):
    """ Mode of the Beta(alpha, beta) distribution"""
    return (alpha - 1) / (alpha + beta - 2)


@numba.jit(nopython=True)
def init_alpha_beta(N, tau, eps=1):
    """
    Initialize the parameters of the prior distributions over 
    the block-specific allele-specific expression ratios.
    
    We put Beta(alpha_{b, t}, beta_{b, t}) prior as it's a conjugate
    prior of a binomial distribution.
    
    We explicitly store alpha and beta parameter matrices
    to make posterior updates easier computationally.
    
    See the manuscript for more details.
    In a nutshell, we select alpha_{b, t} and beta_{b, t}
    in such a way that Beta(alpha_{b, t}, beta_{b, t}) has a mode
    of m / (m + p), where tau[t] = (m, p) — supposed CNV configuration.
    
    :param N: number of blocks
    :param tau: CNV configurations of the form (#maternal, #paternal)
    """
    
    # ASE ratios are stored as (\alpha, \beta) parameter tuples
    # of the underlying Beta distributions
    
    Alpha = np.zeros(shape=(N, tau.shape[0]), dtype=np.float64)
    Beta = np.zeros(shape=(N, tau.shape[0]), dtype=np.float64)

    # See the manuscript for details.
    # Analytical solution is parametric: alpha is linearly dependent from beta
    # The larger alpha and beta are, the stonger are our prior assumptions.
    # Thereby, we select some reasonably small feasible values.
    n_state = tau.shape[0]
    ## generate Theta parameter for ASR; beta distribution
    base = 0.1  # avoiding theta to be 0 or 1
    scale = 30.0  # control the precision of theta

    for config_id in range(n_state):
        alpha, beta = (tau[config_id, :] + base) * scale
        # Theta[:, j] = np.random.beta(_alpha, _beta, size=n_block)
        Alpha[:, config_id] = alpha
        Beta[:, config_id] =  beta
        # k0, k1 = tau[config_id]
        # t = k0 + k1
        #
        # if k0 == 0:
        #     # maternal copy is absent
        #     alpha, beta = 1, 1 + eps
        # elif k1 == 0:
        #     # paternal copy is missing
        #     alpha, beta = 1 + eps, 1
        # else:
        #     if k1 > k0:
        #         # more paternal copies
        #         # distribution should be skewed to the left
        #         # meaning that alpha should be smaller than beta
        #         alpha = 1 + eps
        #         beta = k1 / k0 * alpha + (k0 - k1) / k0
        #     else:
        #         # more maternal copies
        #         # = more probability mass on the right
        #         # meaning that alpha should be larger than beta
        #         beta = 1 + eps
        #         alpha = k0 / k1 * beta + (k1 - k0) / k1
        #         assert beta_mode(alpha, beta) == k0 / t
        #     assert alpha >= 1 and beta >= 1
        # config_id = config_to_id(n_total=t, n_paternal=k1)
        # Alpha[:, config_id] = alpha
        # Beta[:, config_id] = beta
    return Alpha, Beta


""" TODO: DEPRECATE """
@numba.jit(nopython=True)
def init_H_X(N, M, T, I, Theta):
    """
    Precompute H_*, X_* matrices.
    See manuscript for more details.
    
    :param N: number of blocks (rows)
    :param M: number of cells (columns)
    :param I: cell-to-clone assignment
    :param Theta: current ASR matrix
    """
    H = np.full((N, M), np.nan, dtype=np.float64)
    X = np.full((N, M), np.nan, dtype=np.float64)

    for cell_id in range(M):
        for block_id in range(N):
            H[block_id, cell_id] = T[block_id, I[cell_id]]
            if ~np.isnan(H[block_id, cell_id]):
                X[block_id, cell_id] = Theta[
                    block_id, 
                    int(H[block_id, cell_id])
                ]
    return H, X

@numba.jit(nopython=True)
def compute_assignment_probas(cell_id, A, R,  logbincoeffs, X_CLONE, f):
    K = X_CLONE.shape[1]
    # In the manuscript we use the probabilities,
    # but in the code we are forced to use logits.
    logits = np.array([
        cell_loglikelihood(
            A=A[:, cell_id],
            R=R[:, cell_id],
            X=X_CLONE[:, clone_id],
            logbincoeffs=logbincoeffs[:, cell_id]
        )
        + np.log(f[clone_id])
        for clone_id in np.arange(K)
    ])
    # min-max normalization
    logits_0_1 = (logits - logits.min()) / (logits.max() - logits.min())
    # softmax
    probas = np.exp(logits_0_1) / np.sum(np.exp(logits_0_1))
    return probas


@numba.jit(nopython=True)
def predict_cell_label(cell_id, A, R, logbincoeffs, X_CLONE, f):
    """
    Compute log-likelihood of observing a particular cell
    given clonal ASR profiles. Convert this vector of log-likelihoods
    into a probability distribution (min-max normalization + softmax)
    and sample a label for this particular cell accordingly.
    
    :param cell_id: cell identifier
    :param A: alternative allele count matrix, N x M
    :param R: reference allele count matrix, N x M
    :param D: total count matrix, N x M
    :param X_CLONE: clonal block-specific CNV configurations, scDNA
    :param f: clonal frequencies, scDNA
    """
    probas = compute_assignment_probas(cell_id, A=A, R=R, logbincoeffs=logbincoeffs, X_CLONE=X_CLONE, f=f)
    # select label in accordance with estimated likelihoods
    return rand_choice_nb(np.arange(probas.size), probas)

@numba.jit(nopython=True)
def update_I(A, R, logbincoeffs, X_CLONE, f):
    """
    Recompute the clonal label assignment for scRNA dataset.
    
    :param A: alternative allele count matrix, scRNA, N x M
    :param R: reference allele count matrix, scRNA, N x M
    :param D: total count matrix, scRNA, N x M
    :param X_CLONE: clonal block-specific CNV configurations, scDNA
    :param f: clonal frequencies, scDNA
    """
    
    M = A.shape[1]
    return np.array([
        predict_cell_label(cell_id, A, R, logbincoeffs, X_CLONE, f)
        for cell_id in range(M)
    ])


@numba.jit(nopython=True, parallel=True)
def update_alpha_beta(tau, Alpha, Beta, A, R, H):
    """
    Update ASR posterior. See manuscript for more details.
    Only update those parameters that are affected by cells
    whose clonal label has changed since the last update.
    
    :param tau: CNV configurations of the form (#maternal, #paternal)
    :param Theta: ASR specific for each block in each particular CNV configuration
    :param Alpha: alpha parameters of Beta(alpha, beta) posterior
    :param Beta: beta parameters of Beta(alpha, beta) posterior
    :param A_G: alternative allele count matrix, N x M
    :param R_G: reference allele count matrix, N x M
    :param H_G: cell-specific CNV profiles, N x M
    """
    
    n_state = tau.shape[0]
    new_Alpha = np.full_like(Alpha, np.nan)
    new_Beta = np.full_like(Beta, np.nan)

    # process all of the blocks simultaneously
    # in a vectorized fashion

    for config_id in range(n_state):
        
        # mask all the blocks that are supposedly
        # in this particular CNV configuration
        
        H_mask = (H == config_id)
        
        # parameter updates (see the manuscript)
        
        us = np.sum(np.multiply(A,  H_mask), axis=1)
        vs = np.sum(np.multiply(R, H_mask), axis=1)

        new_Alpha[:, config_id] = Alpha[:, config_id] + us
        new_Beta[:, config_id] = Beta[:, config_id] + vs

    return new_Alpha, new_Beta


def aggregate_by_clone(mx, labels, agg_fn):
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


def simulate_G_T(tau, T_mat, D_RNA_seed, D_DNA_seed, n_cell, random_seed=None):
        """
        First version, not supporting CNV=0.
        This function was written by Yuanhua in the end of October 2019.
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        n_state = tau.shape[0]
        n_block = T_mat.shape[0]
        n_clone = T_mat.shape[1]

        ## generate cell cluster labels; uniform distribution
        I_RNA = np.random.choice(n_clone, size=n_cell)
        I_DNA = np.random.choice(n_clone, size=n_cell)

        ## generate Theta parameter for ASR; beta distribution
        _base = 0.1  # avoiding theta to be 0 or 1
        _scale = 30.0 # control the precision of theta
        
        Alpha = np.zeros((n_block, n_state))
        Beta = np.zeros((n_block, n_state))
        Theta = np.zeros((n_block, n_state))

        ## TODO: That's different from how I sample Thetas
        ## Is it critical for the model's quality?
        for j in range(n_state):
            _alpha, _beta = (tau[j, :] + _base) * _scale
            Theta[:, j] = np.random.beta(_alpha, _beta, size=n_block)
            Alpha[:, j] = _alpha
            Beta[:, j] =  _beta 

        ## Generate DP matrix: uniform distribution, profile reserved
        idx_D_RNA = np.random.choice(D_RNA_seed.shape[1], size=n_cell)
        idx_D_DNA = np.random.choice(D_DNA_seed.shape[1], size=n_cell)

        D_RNA = D_RNA_seed[:, idx_D_RNA].astype(int)
        D_DNA = D_DNA_seed[:, idx_D_DNA].astype(int)

        ## Generate X and AD matrices: binomial distribution
        X_RNA = np.zeros(D_RNA.shape)
        H_RNA = np.zeros(D_RNA.shape)
        for i in range(n_block):
            for j in range(n_cell):
                H_RNA[i, j] = int(T_mat[i, I_RNA[j]])
                X_RNA[i, j] = Theta[i, int(T_mat[i, I_RNA[j]])]
        A_RNA = np.random.binomial(D_RNA, X_RNA)
        R_RNA = D_RNA - A_RNA

        H_DNA = np.zeros(D_DNA.shape)
        X_DNA = np.zeros(D_DNA.shape)
        for i in range(n_block):
            for j in range(n_cell):
                H_DNA[i, j] = int(T_mat[i, I_DNA[j]])
                X_DNA[i, j] = Theta[i, int(T_mat[i, I_DNA[j]])]
        A_DNA = np.random.binomial(D_DNA, X_DNA)
        R_DNA = D_DNA - A_DNA

        colsum_fn = lambda mx: np.sum(mx, axis=1)
        A_CLONE = aggregate_by_clone(
            A_DNA,
            I_DNA,
            colsum_fn
        )
        D_CLONE = aggregate_by_clone(
            D_DNA,
            I_DNA,
            colsum_fn
        )
        R_CLONE = D_CLONE - A_CLONE

        clones, label_counts = np.unique(I_DNA, return_counts=True)
        f = label_counts / n_cell

        logbincoeff_DNA = (
            logfact(D_DNA)
            - logfact(A_DNA)
            - logfact(R_DNA)
        )

        logbincoeff_RNA = (
            logfact(D_RNA)
            - logfact(A_RNA)
            - logfact(R_RNA)
        )

        ## return values
        RV = {
            "N" : n_block,
            "M_RNA" : n_cell,
            "M_DNA" :  n_cell,
            "K" : n_clone,
            "f" : f,

            "tau" : tau,
            "Alpha" : Alpha,
            "Beta" : Beta,
            "Theta" : Theta,

            "T" : T_mat,

            "I_DNA": I_DNA,
            "I_RNA" : I_RNA,

            "X_DNA": X_DNA,
            "X_RNA" : X_RNA,

            "H_DNA" : H_DNA,
            "H_RNA": H_RNA,

            "D_DNA" : D_DNA,
            "D_RNA": D_RNA,

            "A_DNA" : A_DNA,
            "A_RNA": A_RNA,

            "R_DNA": R_DNA,
            "R_RNA" : R_RNA,

            "D_CLONE" : D_CLONE,
            "A_CLONE" : A_CLONE,
            "R_CLONE" : R_CLONE,

            "logbincoeff_DNA" : logbincoeff_DNA,
            "logbincoeff_RNA" : logbincoeff_RNA
        }
        return RV