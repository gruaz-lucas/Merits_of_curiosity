import numpy as np

# ------------------------------------------------------------------------------
# Functions used for the Blahut-Arimoto algorithm
# See https://en.wikipedia.org/wiki/Blahut-Arimoto_algorithm
# ------------------------------------------------------------------------------
    
def func_blahut_arimoto(P_YX, thresh=1e-6, max_iter=100, pass_all=False):
    """
    Blahut-Arimoto algorithm to compute the capacity of a discrete memoryless channel.

    Parameters:
    - P_YX: numpy array of shape (N, M), conditional probabilities P(y|x)
            In RL, P_YX[i, j] = P(s' = j | s, a = i)
    - thresh: convergence threshold
    - max_iter: maximum number of iterations
    - pass_all: if True, keep all iterations of r and c

    Returns:
    - r: if pass_all is False, numpy array of shape (N,), prior probabilities of x (actions)
         if pass_all is True, list of numpy arrays containing r at each iteration
    - c: if pass_all is False, scalar capacity value
         if pass_all is True, list of capacity values at each iteration
    """
    N, M = P_YX.shape

    # Initialize probabilities and auxiliary variables
    r = np.ones(N) / N
    r_list = [r.copy()] if pass_all else None
    c_list = [0.0]

    qi = np.zeros((M, N))
    logr = np.zeros(N)

    for _ in range(max_iter):
        # Update qi
        qi = r * P_YX.T
        row_sums = np.sum(qi, axis=1, keepdims=True)
        nonzero_mask = row_sums.squeeze() != 0  # Correctly reduce dimensions for masking

        # Divide only where row_sums are non-zero, use broadcasting
        qi[nonzero_mask] /= row_sums[nonzero_mask]

        # Set uniform distribution where row_sums is zero
        qi[~nonzero_mask] = 1.0 / N

        # Update ri using a more vectorized approach
        ri = np.prod(np.power(qi.T, P_YX) * (qi.T != 0), axis=1)
        ri /= np.sum(ri)

        # Compute tolerance
        tolerance = np.linalg.norm(ri - r) # L2 norm

        # Handling results for each iteration
        if pass_all:
            r_list.append(ri.copy())
            c_list.append(func_capacity(P_YX, ri, qi))
        else:
            r = ri

        # Compute capacity c
        if not pass_all:
            with np.errstate(divide='ignore', invalid='ignore'):
                logr = np.sum(P_YX * np.log(qi.T / ri[:, np.newaxis]), axis=1)
            c_list[0] = np.sum(ri * logr)

        # Check for convergence
        r = ri
        if tolerance < thresh:
            break

    if pass_all:
        return r_list, c_list
    else:
        return r, c_list[0]

def func_capacity(P_YX, r, q):
    """
    Compute the capacity given P_YX, r, and q.

    Parameters:
    - P_YX: numpy array of shape (N, M)
    - r: numpy array of shape (N,)
    - q: numpy array of shape (M, N)

    Returns:
    - c: capacity value (scalar)
    """
    N, M = P_YX.shape
    logr = np.zeros(N)
    for n in range(N):
        inds = P_YX[n, :] != 0
        P_YX_n_inds = P_YX[n, inds]
        q_inds_n = q[inds, n]
        logr[n] = np.sum(P_YX_n_inds * np.log(q_inds_n / r[n]))
    c = np.sum(r * logr)
   
    return c

# ------------------------------------------------------------------------------
# Empowerment
# ------------------------------------------------------------------------------
def func_R_sas_Empow(agent, sp, test_if_sp_dep=False, test_if_limited_2_sa=False, thresh=1e-10, max_iter=100):
    """
    Compute the empowerment of state sp as the intrinsic reward of transition for (s, a) → sp.

    Parameters:
    - agent: object that must have attributes 'n_actions_per_state' and 'T'
    - sp: integer, the index of the state s'
    - test_if_sp_dep: if True, returns True to indicate the reward depends on sp
    - test_if_limited_2_sa: if True, returns True to indicate the reward should be updated only for (s_t, a_t)
    - thresh: convergence threshold for Blahut-Arimoto algorithm
    - max_iter: maximum number of iterations for Blahut-Arimoto algorithm

    Returns:
    - c: capacity value (scalar), representing the empowerment of state sp
    """
    if test_if_sp_dep:
        # Indicating whether the reward function depends on sp
        return True
    if test_if_limited_2_sa:
        # Indicating whether the reward function should be updated only for (s_t, a_t)
        return True

    n_actions_sp = agent.n_actions_per_state[sp]
    # Extract the conditional probabilities P(s' | s=sp, a)
    # P_YX has shape (N, M), where N is number of actions at state sp, M is number of states
    P_YX = agent.T[sp, :n_actions_sp, :]

    # Compute the capacity using Blahut-Arimoto algorithm
    r, c = func_blahut_arimoto(P_YX, thresh=thresh, max_iter=max_iter, pass_all=False)
    
    return c  # Return the capacity as the empowerment