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
        n * np.log(n) 
        - n 
        + np.log(n * (1 + 4 * n * (1 + 2 * n))) / 6 
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
def cell_logits(cell_id, A, R, D, X):
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
    bincoeffs = logfact(D) - logfact(A) - logfact(R)
    logits = bincoeffs + A * np.log(X) + R * np.log(1 - X)
    return logits 


@numba.jit(nopython=True)
def cell_loglikelihood(cell_id, A, R, D, X):
    """
    Log-likelihood of sampling cell 'cell_id' from profile 'X'. 
    
    :param cell_id: id of the cell of interest, int
    :param A: alternative allele count vector, np.array
    :param R: reference allele count vector, np.array
    :param D: A + R, np.array
    :param X: clonal ASR profile, np.array
    
    """
    logits = cell_logits(cell_id, A, R, D, X)
    return np.sum(logits[np.isfinite(logits)])


# TODO: write tests that compare the output with the naive implementation
@numba.jit(nopython=True)
def total_loglikelihood(A_G_prime, R_G_prime, X_G_prime, bincoeff_prime, 
                        A_G, R_G, X_G, bincoeff):
    """
    Log-likelihood of observing cells from both modalities
    under current clonal label assignment. 
    
    All the input matrices are expected to be np.ndarrays.
    
    :param A_G_prime: alternative allele count vector, scDNA, N_G x M_prime
    :param R_G_prime: reference allele count vector, scDNA, N_G x M_prime
    :param X_G_prime: clonal ASR profiles, scDNA, N_G x M_prime
    :param bincoeff_prime: logarithms of coefficients in binomial model, scDNA, N_G x M_prime
    
    :param A_G: alternative allele count vector, scRNA, N_G x M 
    :param R_G: reference allele count vector, scRNA, N_G x M
    :param X_G: clonal ASR profiles, scRNA, N_G x M
    :param bincoeff: logarithms of coefficients in binomial model, scRNA, N_G x M
    """
    alt_prime = A_G_prime * np.log(X_G_prime)
    ref_prime = R_G_prime * np.log(1 - X_G_prime)
    
    alt = A_G * np.log(X_G)
    ref = R_G * np.log(1 - X_G)
    
    loglik_prime = np.ravel(alt_prime + ref_prime + bincoeff_prime)
    loglik = np.ravel(alt + ref + bincoeff)

    return (
        np.sum(loglik_prime[np.isfinite(loglik_prime)])
        + np.sum(loglik[np.isfinite(loglik)])
    )


def total_loglikelihood_scipy(A_G_prime, R_G_prime, X_G_prime, A_G, R_G, X_G):
    """
    Same as total_likelihood but in pure scipy (and thereby not JIT-compiled).
    This function can be used as a sanity check. 
    
    TODO: Actually, I need to WRITE TESTS that ensure that results 
    of these two functions agree.
    """
    
    M_prime, M = A_G_prime.shape[1], A_G.shape[1]
    logits_prime = sps.binom(
        n=D_G_prime, 
        p=X_G_prime
    ).logpmf(A_G_prime)
    logits = sps.binom(
        n=D_G, 
        p=X_G
    ).logpmf(A_G)
    return np.sum(logits_prime[np.isfinite(logits_prime)]) \
            + np.sum(logits[np.isfinite(logits)])
#     return (
#         np.sum([
#             cell_loglikelihood(cell_id, A_G_prime[:, cell_id], D_G_prime[:, cell_id], X_G_prime[:, cell_id])
#             for cell_id in range(M_prime)
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
def init_T(A_C_prime, D_C_prime, CNV_prime, N_G, K):
    """
    Initializes T — clone-specific CNV profiles.
    The problem is, we only have total CNV numbers, not 
    (#maternal, #paternal) tuples (configuration, (m, p) below). 
    
    We have to estimate the configuration for each block in each clone.
    We do so by choosing the (m, p) tuple such that m / (m + p) is 
    as close as possible to the actual observed ASR: 
    
    (m, p) := \argmin_{x, y} | AD[block, clone] / DP[block, clone] - x/(x+y) |
    
    :param A_C_prime: allele-specific count matrix, scDNA, N_G x K
    :param D_C_prime: total count matrix, scDNA, N_G x K
    :param CNV_prime: raw CNV profiles for individual cells, scDNA
    :param N_G: number of haplotype blocks
    :param K: numbe of clones
    """
    
    T = np.full((N_G, K), np.nan, dtype=np.float64)
    for clone_id in range(K):
        for block_id in range(N_G):
            t = CNV_prime[block_id, clone_id]
            ad = A_C_prime[block_id, clone_id]
            dp = D_C_prime[block_id, clone_id]
            if dp == 0 or dp is np.nan:
                continue
            ase_ratio = ad / dp # MLE estimate
            # np.arange(t + 1) / t = (0 / t, 1 / t, ..., t / t)
            # TODO: make it a separate function?
            n_maternal = np.argmin(np.abs(
                np.arange(t + 1)[::-1] / t 
                - ase_ratio
            ))
            n_paternal = t - n_maternal
            # ((0, 1), (1, 0)), ..., ((0, t), ..., (t, 0))
            # CNV number t generates exactly t different configurations
            # degenerate configuration (0, 0) is excluded, hence -1
            T[block_id, clone_id] = config_to_id(t, n_paternal)
    return T


@numba.jit(nopython=True)
def init_alpha_beta(N_G, tau):
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
    
    :param N_G: number of blocks
    :param tau: CNV configurations of the form (#maternal, #paternal)
    """
    
    # ASE ratios are stored as (\alpha, \beta) parameter tuples
    # of the underlying Beta distributions
    
    Alpha_G = np.zeros(shape=(N_G, tau.shape[0]), dtype=np.float64)
    Beta_G = np.zeros(shape=(N_G, tau.shape[0]), dtype=np.float64)

    # See the manuscript for details.
    # Analytical solution is parametric: alpha is linearly dependent from beta
    # The larger alpha and beta are, the stonger are our prior assumptions.
    # Therbey, we select some reasonably small feasible values.
    
    eps = 1
    for config_id in range(tau.shape[0]):
#         for block_id in range(N_G):

        k0, k1 = tau[config_id]
        t = k0 + k1

        if k0 == 0:
            # maternal copy is absent
            alpha, beta = 1, 1 + eps
        elif k1 == 0:
            # paternal copy is missing
            alpha, beta = 1 + eps, 1
        else:
            # more paternal copies
            # distribution should be skewed to the right
            if k1 > k0:
                alpha = 1 + eps
                beta = k1 / k0 * alpha + (k0 - k1) / k0
            # more maternal copies
            # = more probability mass on the left
            else:
                beta = 1 + eps
                alpha = k0 / k1 * beta + (k1 - k0) / k1
                # Can't use assertions with numba.jit

#                 assert np.isclose(beta_mode(alpha, beta), k0 / t)
#             assert alpha >= 1 and beta >= 1
        config_id = config_to_id(t, k1)
        Alpha_G[:, config_id] = alpha
        Beta_G[:, config_id] = beta
    return Alpha_G, Beta_G


@numba.jit(nopython=True)
def init_H_X(N, M, T, I, Theta_G):
    """
    Precompute H_*, X_* matrices.
    See manuscript for more details.
    
    :param N: number of blocks (rows)
    :param M: number of cells (columns)
    :param I: cell-to-clone assignment
    :param Theta_G: current ASR matrix
    """
    H = np.full((N, M), np.nan, dtype=np.float64)
    X = np.full((N, M), np.nan, dtype=np.float64)

    for cell_id in range(M):
        for block_id in range(N):
            H[block_id, cell_id] = T[block_id, I[cell_id]]
            if ~np.isnan(H[block_id, cell_id]):
                X[block_id, cell_id] = Theta_G[
                    block_id, 
                    int(H[block_id, cell_id])
                ]
    return H, X


@numba.jit(nopython=True)
def predict_cell_label(cell_id, A, R, D, X_C_prime, f):
    """
    Compute log-likelihood of observing a particular cell
    given clonal ASR profiles. Convert this vector of log-likelihoods
    into a probability distribution (min-max normalization + softmax)
    and sample a label for this particular cell accordingly.
    
    :param cell_id: cell identifier
    :param A: alternative allele count matrix, N x M
    :param R: reference allele count matrix, N x M
    :param D: total count matrix, N x M
    :param X_C_prime: clonal block-specific CNV configurations, scDNA
    :param f: clonal frequencies, scDNA
    """
    K = X_C_prime.shape[1]    
    # In the manuscript we use the probabilities,
    # but in the code we are forced to use logits.
    logits = np.array([
        cell_loglikelihood(
            cell_id, 
            A[:, cell_id],
            R[:, cell_id],
            D[:, cell_id], 
            X_C_prime[:, clone_id]
        ) 
        + np.log(f[clone_id])
        for clone_id in np.arange(K)
    ])
    # min-max normalization
    logits_normalized = (logits - logits.min()) / (logits.max() - logits.min())
    # softmax
    probas = np.exp(logits_normalized) / np.sum(np.exp(logits_normalized))
    # select label in accordance with estimated likelihoods
    return rand_choice_nb(np.arange(K), probas)

@numba.jit(nopython=True)
def update_I_G(A, R, D, X_C_prime, f):
    """
    Recompute the clonal label assignment for scRNA dataset.
    
    :param A: alternative allele count matrix, scRNA, N_G x M
    :param R: reference allele count matrix, scRNA, N_G x M
    :param D: total count matrix, scRNA, N_G x M
    :param X_C_prime: clonal block-specific CNV configurations, scDNA
    :param f: clonal frequencies, scDNA
    """
    
    M = A.shape[1]
    return np.array([
        predict_cell_label(cell_id, A, R, D, X_C_prime, f)
        for cell_id in range(M)
    ])


@numba.jit(nopython=True, parallel=True)
def update_alpha_beta(tau, Theta_G, Alpha_G, Beta_G, A_G, D_G, H_G, changed_mask):
    """
    Update ASR posterior. See manuscript for more details.
    Only update those parameters that are affected by cells
    whose clonal label has changed since the last update.
    
    :param tau: CNV configurations of the form (#maternal, #paternal)
    :param Theta_G: ASR specific for each block in each particular CNV configuration 
    :param Alpha_G: alpha parameters of Beta(alpha, beta) posterior
    :param Beta_G: beta parameters of Beta(alpha, beta) posterior
    :param A_G: alternative allele count matrix, N x M
    :param D_G: total count matrix, N x M
    :param H_G: cell-specific ASR profiles according to the current label assignment
    :param changed_mask: boolean mask showing which labels have changed since last update
    """
    
    N_G = Alpha_G.shape[0]
    new_Alpha_G = np.full_like(Alpha_G, np.nan)
    new_Beta_G = np.full_like(Beta_G, np.nan)

    A_changed = A_G[:, changed_mask]
    D_changed = D_G[:, changed_mask]
    R_changed = D_changed - A_changed
    H_changed = D_G[:, changed_mask]

    for config_id in range(tau.shape[0]):
        
        # mask all the blocks that are supposedly
        # in this particular CNV configuration
        
        h_mask = H_changed == config_id
        
        # process all of the blocks simultaneously
        # in a vectorized fashion
        
        alphas = Alpha_G[:, config_id]
        betas = Alpha_G[:, config_id]
        
        # parameter updates (see the manuscript)
        
        us = np.sum(A_changed * h_mask, axis=1)
        vs = np.sum(R_changed * h_mask, axis=1)

        new_Alpha_G[:, config_id] = alphas + us
        new_Beta_G[:, config_id] = betas + vs

    return new_Alpha_G, new_Beta_G