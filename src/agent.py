import numpy as np
import random
from .empow_funcs import func_R_sas_Empow

class ModelBasedAgent():
    """
    Base class for Model-Based Agents.
    Subclasses must implement the compute_intrinsic_reward method.
    """
    
    def __init__(self, n_states, n_actions_per_state, ε, βi, λi, βe, λe, T_PS=None, model_fixed=False):
        # Store the environment information and hyperparameters
        self.n_states = n_states                            # Number of states
        self.n_actions_per_state = n_actions_per_state      # Number of actions per state
        self.max_actions = np.max(n_actions_per_state)      # Maximum number of actions
        self.ε = ε                                          # Small constant to avoid unseen transitions to be assigned 0 probability
        self.βi = βi                                        # Inverse temperature for intrinsic reward
        self.λi = λi                                        # Discount factor for intrinsic reward
        self.βe = βe                                        # Inverse temperature for extrinsic reward
        self.λe = λe                                        # Discount factor for extrinsic reward
        self.T_PS = T_PS if T_PS is not None else n_states  # Number of iterations for prioritized sweeping
        self.model_fixed = model_fixed                      # Boolean indicating whether the model is fixed to the ground truth
        
        # Initialize the transition and reward models 
        self.T = np.full((self.n_states, self.max_actions, self.n_states), 1.0 / self.n_states) # Transition model
        self.Ri = np.zeros((self.n_states, self.max_actions, self.n_states))                    # Intrinsic reward
        self.Re = np.zeros((self.n_states, self.max_actions, self.n_states))                    # Extrinsic reward
        self.CumRe = np.zeros((self.n_states, self.max_actions, self.n_states))                 # Cumulative extrinsic reward
        
        # Initialize the value function
        self.Qi = np.zeros((self.n_states, self.max_actions)) # Q-values for intrinsic reward
        self.Qe = np.zeros((self.n_states, self.max_actions)) # Q-values for extrinsic reward
        self.Ui = np.zeros(self.n_states) # State values = max of Q-values
        self.Ue = np.zeros(self.n_states) # State values = max of Q-values
        
        # Initialize the state visitation counts
        self.Csas = np.zeros((self.n_states, self.max_actions, self.n_states))  # Counts of state-action-state transitions
        self.Csa = np.zeros((self.n_states, self.max_actions))                  # Counts of state-action transitions
        self.Cs = np.zeros(self.n_states)                                       # Counts of state visits
    
    def compute_policy(self, state):
        # Compute the policy as softmax on the Q-values, with inverse temperature β
        actions = self.n_actions_per_state[state]
        Q = self.βi * self.Qi[state, :actions] + self.βe * self.Qe[state, :actions]
        Q -= np.min(Q)  # Normalize Q for numerical stability
        exp_Q = np.exp(Q)
        π = exp_Q / np.sum(exp_Q)
        if np.any(np.isnan(π)):
            # if exp overflows, set to 1.0 and normalize
            π = np.nan_to_num(π, nan=1.0)
            π /= np.sum(π)
        return π
        
    def sample_action(self, state):
        # Compute the policy
        π = self.compute_policy(state)
        n_actions = self.n_actions_per_state[state]
        assert len(π) == n_actions, f"Policy length {len(π)} does not match number of actions {n_actions}"
        return np.random.choice(n_actions, p=π)
    
    def update_model(self, state, action, next_state, reward):
        # Update the transition model
        self.Csas[state, action, next_state] += 1
        self.Csa[state, action] += 1
        self.Cs[state] += 1
        
        if not self.model_fixed:
            α_sas = self.Csas + self.ε
            self.T = α_sas / np.sum(α_sas, axis=2, keepdims=True)            
        
        # Update the extrinsic reward model
        self.CumRe[state, action, next_state] += reward
        self.Re[state, action, next_state] = self.CumRe[state, action, next_state] / self.Csas[state, action, next_state]
        
        # Update the intrinsic reward model
        self.compute_intrinsic_reward()
        self.prioritized_sweeping(is_intrinsic=True)

    
    def prioritized_sweeping(self, is_intrinsic, ΔV_thresh=1e-4, theta_thresh=1e-4):
        """
        Prioritized Sweeping Algorithm to update Q and U values for the agent.

        Parameters:
        - is_intrinsic: Boolean indicating whether to update intrinsic or extrinsic values.
        - ΔV_thresh: Threshold for stopping updates based on ΔV.
        - theta_thresh: Threshold for determining significant changes in Q-values.
        """
         
        if is_intrinsic:
            λ = self.λi
            Q = self.Qi
            U = self.Ui
            R = self.Ri
        else:
            λ = self.λe
            Q = self.Qe
            U = self.Ue
            R = self.Re
        
        # Compute R + (λ * U) in the correct format
        u_reshaped = np.tile(U, (self.n_states, self.max_actions, 1)) # u_reshaped[s, a, s'] = U(s')
        r_plus_u = R + λ * u_reshaped # r_plus_u[s, a, s'] = R(s, a, s') + λ * U(s')

        # Update Q-values
        Q[:] = np.sum(self.T * r_plus_u, axis=2) # Q(s, a) = Σ_s' T(s, a, s') * (R(s, a, s') + λ * U(s'))

        V = np.zeros(self.n_states)
        priorities = np.zeros(self.n_states)

        # Compute V-values and priorities
        for s in range(self.n_states):
            V[s] = np.max(Q[s, :self.n_actions_per_state[s]])  # Max Q-value
            priorities[s] = abs(U[s] - V[s])

        # Update U-values in T_PS steps
        for _ in range(self.T_PS):
            s_prime = np.argmax(priorities)
            ΔV = V[s_prime] - U[s_prime]
            
            # Check stopping condition
            if abs(ΔV) / abs(np.max(V) - np.min(V)) <= ΔV_thresh:
                break

            U[s_prime] = V[s_prime] # Update U-value

            # Update Q-values and priorities
            mask = self.T[:, :, s_prime] * np.abs(ΔV) > theta_thresh # Only update Q-values for significant changes
            for s in np.where(mask.any(axis=1))[0]:
                Q[s, :self.n_actions_per_state[s]] += λ * self.T[s, :self.n_actions_per_state[s], s_prime] * ΔV
                V[s] = np.max(Q[s, :self.n_actions_per_state[s]])  # V = max_a Q(s, a)
                priorities[s] = abs(U[s] - V[s]) # Update priorities

        # Save the updated Q-values (and optionally U-values) back to the agent
        if is_intrinsic:
            self.Qi = Q
            self.Ui = U
        else:
            self.Qe = Q
            self.Ue = U
        return
    
    def reset_model(self):
        # Reset the transition and reward models
        self.T.fill(1.0 / self.n_states)
        self.Ri.fill(0.0)
        self.Re.fill(0.0)
        self.CumRe.fill(0.0)
        
        # Reset the value functions
        self.Qi.fill(0.0)
        self.Qe.fill(0.0)
        self.Ui.fill(0.0)
        self.Ue.fill(0.0)
        
        # Reset the state visitation counts
        self.Csas.fill(0.0)
        self.Csa.fill(0.0)
        self.Cs.fill(0.0)
    
    def compute_intrinsic_reward(self):
        # Compute the intrinsic reward for each transition based on internal models
        # i.e. update self.Ri
        raise NotImplementedError("Subclasses must implement this method.")
    
class RandomAgent(ModelBasedAgent):
    """
    Agent with random policy and no intrinsic reward.
    """
    
    def compute_intrinsic_reward(self):
        # Zero-out the intrinsic reward matrix
        if np.any(self.Ri != 0.0):
            self.Ri.fill(0.0)
    
class NoveltyAgent(ModelBasedAgent):
    """
    Agent with Novelty as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        # Compute novelty for each state s
        t = np.sum(self.Cs)
        nov_s = - np.log((self.Cs + self.ε) / (t + self.ε * self.n_states))  # Formula for novelty of each state
        
        # Repeat novelty to match the dimensions of (s, a, s') for each state-action-state transition
        self.Ri = np.tile(nov_s, (self.n_states, self.max_actions, 1))
        
        

class SurpriseAgent(ModelBasedAgent):
    """
    Agent with Surprise as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        sur = -np.log(self.T)  # Surprise[s, a, s'] = -log(T(s, a, s'))
        
        # Handle Inf values if the model is fixed (some transitions have probability 0)
        if self.model_fixed and np.any(np.isinf(sur)):
            sur = np.nan_to_num(sur, nan=0.0)  # Replace Inf with 0.0
        
        self.Ri = sur
   
    
class InformationGainAgent(ModelBasedAgent):
    """
    Agent with Information Gain as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        if self.model_fixed:
            self.Ri.fill(0.0)  # Zero-out the reward matrix if the model is fixed
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions_per_state[s]):
                    # Compute Information Gain for each state-action-state transition
                    α_sa = self.ε * self.n_states + self.Csa[s, a]
                    α_sas = self.ε + self.Csas[s, a, :]
                    self.Ri[s, a, :] = (
                        np.log((α_sa + 1) / α_sa)
                        + (α_sas / α_sa) * np.log(α_sas / (α_sas + 1))
                    )

                   
class EmpowermentAgent(ModelBasedAgent):
    """
    Agent with Empowerment as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        for s_prime in range(self.n_states):
            empow = func_R_sas_Empow(self, s_prime)  # Compute empowerment for s'
            empow = 0.0 if abs(empow) < np.finfo(float).eps else empow  # Round to zero if very small
            self.Ri[:, :, s_prime] = empow
    
class SPIEAgent(ModelBasedAgent):
    """
    Agent with implementing Successor-Predecessor Intrinsic Exploration.
    (Yu et al., 2024, https://arxiv.org/abs/2305.15277)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the Successor-Predecessor matrix
        self.sp_M = np.zeros((self.n_states, self.n_states))
        self.sp_TPS = self.n_states # Number of iterations for prioritized sweeping
        
    def compute_intrinsic_reward(self):
        
        self.sp_M = self.update_sp() # Update the Successor-Predecessor matrix

        retro = np.sum(np.abs(self.sp_M), axis=0) 
        self.Ri = self.sp_M[:, None, :] - retro # Ri[s, a, s'] = M[s, s'] - Σ_s'' M[s'', s']

    def update_sp(self, ΔV_thresh=1e-4, theta_thresh=1e-4):
        # Update the Successor-Predecessor matrix using Prioritized Sweeping
        n = self.n_states
        P = np.zeros((n, n)) # Transition matrix for the policy
        
        # Compute policy and populate P matrix
        for s in range(n):
            pis = self.compute_policy(s)
            P[s] = np.sum(pis[:, None] * self.T[s, :self.n_actions_per_state[s], :], axis=0)
        
        # Initialize variables
        λ = self.λi
        M_old = np.copy(self.sp_M)
        
        # Update M-values
        M = np.eye(n) + λ * P @ M_old # M[s,s'] = δ(s,s') + λ * Σ_s' P(s' | s) * M_old[s', s']
        
        # Create the priority queue
        Prior = np.abs(M - M_old)
        
        # Update U-values for T_PS steps
        for _ in range(self.sp_TPS):
            # Find the element with the largest priority
            s3, s2 = np.unravel_index(np.argmax(Prior), Prior.shape)
            delta_M = M[s3, s2] - M_old[s3, s2]
            
            # Check stopping condition
            if abs(delta_M) / abs(np.max(M) - np.min(M)) <= ΔV_thresh:
                break
            
            # Update M_old with new value
            M_old[s3, s2] = M[s3, s2]
            
            # Propagate updates to M and adjust Prior
            for s1 in range(n):
                if theta_thresh == 0 or (P[s1, s3] * abs(delta_M) > theta_thresh):
                    M[s1, s2] += λ * P[s1, s3] * delta_M
                    Prior[s1, s2] = abs(M[s1, s2] - M_old[s1, s2])
        
        return M

class MOPAgent(ModelBasedAgent):
    """
    Agent implementing Maximum Occupancy Principle.
    For details, see: (Ramirez-Ruiz et al., 2024, https://www.nature.com/articles/s41467-024-49711-1)
    """
    
    def __init__(self, *args, mop_α=1.0, mop_β=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the MOP hyperparameters
        self.mop_α = mop_α
        self.mop_β = mop_β
        self.mop_V = np.zeros(self.n_states)
        self.mop_π = np.zeros((self.n_states, self.max_actions))
        # Initialize the policy as uniform
        for s in range(self.n_states):
            self.mop_π[s, :self.n_actions_per_state[s]] = 1 / self.n_actions_per_state[s]
        
    def compute_policy(self, state):
        self.mop_optimize_policy()  # Update the policy
        return self.mop_π[state, :self.n_actions_per_state[state]]
    
    def update_model(self, state, action, next_state, reward):
        # Update the counts, transition and extrinsic reward models
        # but does not call compute_intrinsic_reward and prioritized_sweeping
        self.Csas[state, action, next_state] += 1
        self.Csa[state, action] += 1
        self.Cs[state] += 1
        
        if not self.model_fixed:
            α_sas = self.Csas + self.ε
            self.T[state] = α_sas[state] / np.sum(α_sas[state], axis=1, keepdims=True) 
                       
        # Update the extrinsic reward model
        self.CumRe[state, action, next_state] += reward
        self.Re[state, action, next_state] = self.CumRe[state, action, next_state] / self.Csas[state, action, next_state]
    
    def entropy(self, probs):
        # Compute the entropy of a probability distribution
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    
    def mop_Z(self, state, V, H, α, β, λ):
        # Compute the partition function Z for the MOP optimization
        actions = range(self.n_actions_per_state[state])
        expected_values = np.sum(self.T[state, :self.n_actions_per_state[state], :] * V, axis=1)
        return np.sum(np.exp((1 / α) * (β * H[state, :len(actions)] + λ * expected_values)))
    
    def mop_value_iteration(self, V, H, α, β, λ, max_iters=1000, tol=1e-6):
        # Value iteration for the MOP optimization
        for _ in range(max_iters):
            V_prev = V.copy()
            Z = np.array([self.mop_Z(s, V, H, α, β, λ) for s in range(self.n_states)])  # Vectorized
            V[:] = α * np.log(Z)
            
            if np.max(np.abs(V_prev - V)) < tol:
                break

    def mop_update_policy(self, policy, V, H, α, β, λ):
        # Update the policy based on the current value function
        for s in range(self.n_states):
            Z_s = self.mop_Z(s, V, H, α, β, λ)
            expected_values = np.sum(self.T[s, :self.n_actions_per_state[s], :] * V, axis=1)
            policy[s, :self.n_actions_per_state[s]] = np.exp((1 / α) * (β * H[s, :self.n_actions_per_state[s]] + λ * expected_values)) / Z_s

    def mop_optimize_policy(self, max_iters=500, v_tol=1e-4, π_tol=1e-4):
        # Optimize the policy using a modified version of value iteration (see https://www.nature.com/articles/s41467-024-49711-1)
        α, β, λ, V, π = self.mop_α, self.mop_β, self.λi, self.mop_V, self.mop_π

        H = np.zeros((self.n_states, max(self.n_actions_per_state)))
        for s in range(self.n_states):
            for a in range(self.n_actions_per_state[s]):
                H[s, a] = self.entropy(self.T[s, a, :])
        
        π_old = π.copy()
        for _ in range(max_iters):
            self.mop_value_iteration(V, H, α, β, λ, tol=v_tol)
            self.mop_update_policy(π, V, H, α, β, λ)
            if np.max(np.abs(π_old - π)) < π_tol:
                break
            π_old[:] = π

        # Handle NaNs in the policy
        π[np.isnan(π)] = 1
        π /= np.sum(π, axis=1, keepdims=True)

        self.mop_V, self.mop_π = V, π
        return π
