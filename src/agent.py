import numpy as np
from .empow_funcs import func_R_sas_Empow

class ModelBasedAgent():
    
    def __init__(self, n_states, n_actions_per_state, ε, βi, λi, βe, λe, T_PS=None, model_fixed=False, r0=0.0):
        # Store the environment information and hyperparameters
        self.n_states = n_states
        self.n_actions_per_state = n_actions_per_state
        self.max_actions = np.max(n_actions_per_state)
        self.ε = ε
        self.βi = βi
        self.λi = λi
        self.βe = βe
        self.λe = λe
        self.T_PS = T_PS if T_PS is not None else n_states
        self.model_fixed = model_fixed
        
        # Initialize the transition and reward models 
        self.T = np.full((self.n_states, self.max_actions, self.n_states), 1.0 / self.n_states) # Transition model
        self.Ri = np.zeros((self.n_states, self.max_actions, self.n_states))    # Intrinsic reward
        self.Re = np.full((self.n_states, self.max_actions, self.n_states), r0) # Extrinsic reward
        self.CumRe = np.zeros((self.n_states, self.max_actions, self.n_states)) # Cumulative extrinsic reward
        
        # Initialize the value function
        self.Qi = np.zeros((self.n_states, self.max_actions))
        self.Qe = np.zeros((self.n_states, self.max_actions))
        self.Ui = np.zeros(self.n_states) # State values = max of Q-values
        self.Ue = np.zeros(self.n_states) # State values = max of Q-values
        
        # Initialize the state visitation counts
        self.Csas = np.zeros((self.n_states, self.max_actions, self.n_states))
        self.Csa = np.zeros((self.n_states, self.max_actions))
        self.Cs = np.zeros(self.n_states)
    
    def compute_policy(self, state):
        # Compute the policy as softmax on the Q-values, with inverse temperature β
        actions = self.n_actions_per_state[state]
        Q = self.βi * self.Qi[state, :actions] + self.βe * self.Qe[state, :actions]
        Q -= np.min(Q)  # Normalize Q for numerical stability
        exp_Q = np.exp(Q)
        π = exp_Q / np.sum(exp_Q)
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
            Q = self.Qi  # Assuming the same Q-values are used for intrinsic updates
            U = self.Ui
            R = self.Ri
        else:
            λ = self.λe
            Q = self.Qe  # Assuming the same Q-values are used for extrinsic updates
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
            
            if abs(ΔV) / abs(np.max(V) - np.min(V)) <= ΔV_thresh:
                break

            U[s_prime] = V[s_prime]

            # Update Q-values and priorities
            mask = self.T[:, :, s_prime] * np.abs(ΔV) > theta_thresh
            for s in np.where(mask.any(axis=1))[0]:
                Q[s, :self.n_actions_per_state[s]] += λ * self.T[s, :self.n_actions_per_state[s], s_prime] * ΔV
                V[s] = np.max(Q[s, :self.n_actions_per_state[s]])  # Max Q-value
                priorities[s] = abs(U[s] - V[s])

        # Save the updated Q-values (and optionally U-values) back to the agent
        if is_intrinsic:
            self.Qi = Q
            self.Ui = U
        else:
            self.Qe = Q
            self.Ue = U
        return
    
    def compute_intrinsic_reward(self):
        # Compute the intrinsic reward for each transition based on internal models
        # i.e. update self.Ri
        raise NotImplementedError("Subclasses must implement this method.")
    
class RandomAgent(ModelBasedAgent):
    """
    Agent with random policy and no intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        self.Ri[:,:,:] = 0 # Zero-out the reward matrix
        pass
    
    
class NoveltyAgent(ModelBasedAgent):
    """
    Agent with Novelty as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        # Compute novelty for each state s
        t = np.sum(self.Cs)
        nov_s = - np.log((self.Cs + self.ε) / (t + self.ε * self.n_states))  # Formula for novelty
        
        # Repeat novelty to match the dimensions of (s, a, s') for each state-action-state transition
        self.Ri = np.tile(nov_s, (self.n_states, self.max_actions, 1))
        
        

class SurpriseAgent(ModelBasedAgent):
    """
    Agent with Surprise as intrinsic reward.
    """
    def compute_intrinsic_reward(self):
        sur = -np.log(self.T)  # Surprise[s, a, s'] = -log(T(s, a, s'))
        
        # Handle Inf values if the model is fixed
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
        