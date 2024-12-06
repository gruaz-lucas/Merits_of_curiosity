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
    N, M = P_YX.shape  # N: size of alphabet x (number of actions)
                       # M: size of alphabet y (number of states)

    # Initialization of r (prior policy)
    r_list = [np.ones(N) / N]
    c_list = [0.0]

    # Initialize qi and ri
    qi = np.zeros((M, N))
    ri = np.zeros(N)

    # Auxiliary variables
    inds = np.zeros(M, dtype=bool)
    logr = np.zeros(N)

    for iteration in range(max_iter):
        r = r_list[-1]

        # Update qi
        for m in range(M):
            qi[m, :] = r * P_YX[:, m]
            if np.sum(qi[m, :]) == 0:
                qi[m, :] = 1.0 / N
            else:
                qi[m, :] /= np.sum(qi[m, :])

        # Update ri
        for n in range(N):
            inds = qi[:, n] != 0
            ri_numerator = qi[inds, n] ** P_YX[n, inds]
            ri[n] = np.prod(ri_numerator)

        ri /= np.sum(ri)  # Normalize ri

        # Compute tolerance to check for convergence
        tolerance = np.sum((ri - r) ** 2)

        if pass_all:
            r_list.append(ri.copy())
            c_list.append(func_capacity(P_YX, ri, qi))
        else:
            r_list[0] = ri.copy()

            # Compute capacity c
            for n in range(N):
                inds = P_YX[n, :] != 0
                numerator = qi[inds, n]
                denominator = ri[n]
                logr[n] = np.sum(P_YX[n, inds] * np.log(numerator / denominator))
            c_list[0] = np.sum(ri * logr)

        # Check for convergence
        if tolerance < thresh:
            break

    if pass_all:
        return r_list, c_list
    else:
        return r_list[0], c_list[0]

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

def func_capacity_alternative(P_YX, r):
    """
    Alternative method to compute capacity.

    Parameters:
    - P_YX: numpy array of shape (N, M)
    - r: numpy array of shape (N,)

    Returns:
    - c: capacity value (scalar)
    """
    N, M = P_YX.shape
    P_Y = np.dot(r, P_YX)  # Marginal probability of y
    inds_P_Y = P_Y != 0
    H_Y = -np.sum(P_Y[inds_P_Y] * np.log(P_Y[inds_P_Y]))  # Entropy H(Y)

    H_YX = np.zeros(N)
    for n in range(N):
        P_YX_n = P_YX[n, :]
        inds_P_YX_n = P_YX_n != 0
        H_YX[n] = -np.sum(P_YX_n[inds_P_YX_n] * np.log(P_YX_n[inds_P_YX_n]))  # Entropy H(Y|X=n)
    H_YX_total = np.sum(r * H_YX)  # Weighted sum over r

    c = H_Y - H_YX_total
    return c

# ------------------------------------------------------------------------------
# Empowerment
# ------------------------------------------------------------------------------
def func_R_sas_Empow(agent, sp, test_if_sp_dep=False, test_if_limited_2_sa=False, thresh=1e-6, max_iter=100):
    """
    Compute the empowerment of state sp as the intrinsic reward of transition for (s, a) → sp.

    Parameters:
    - agent: object that must have attributes 'n_actions_per_state' and 'Phat_sa_s'
    - sp: integer, the index of the state sp
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