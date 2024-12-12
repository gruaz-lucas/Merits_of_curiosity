
import numpy as np
from math import sqrt
from .room_environment import RoomEnvironment

def run_steps(ag, env, n_steps):
    # Run the agent in the environment for n_steps steps 
    for _ in range(n_steps):
        state = env.state
        a = ag.sample_action(state)
        obs, reward, done, truncated, info = env.step(a)
        ag.update_model(state, a, obs['state'], reward)
    return ag, env


class Simulator():
    """
    Class to run an agent in an environment and measure performance.
    """
    
    def __init__(self, measure_type, n_iter, eval_every, verbose=False):
        self.measure_type = measure_type    # String: type of performance measure
        self.n_iter = n_iter                # Int: number of iterations to run the agent
        self.eval_every = eval_every        # Int: evaluate performance every eval_every iterations
        self.verbose = verbose              # Boolean: print performance at each evaluation
        
    def evaluate(self, ag, env):
        """
        Evaluate the agent's performance based on the specified measure at the current time step.
        Args:
            ag: Agent object
            env: Environment object
        Returns:
            performance: Scalar value representing the performance at the current time
        """

        if self.measure_type == 'state_discovery':
            # Compute the proportion of unvisited states
            unvisited = np.sum(ag.Cs == 0)
            return unvisited / ag.n_states
        
        elif self.measure_type == 'model_accuracy':
            # Compute RMSE between true and estimated models
            err = 0
            for s in range(ag.n_states):
                for a in range(ag.n_actions_per_state[s]):
                    err += np.sum((ag.T[s, a] - env.transition_matrix[s, a]) ** 2)
            err /= ag.n_states * np.sum(ag.n_actions_per_state)
            return sqrt(err)
        
        elif self.measure_type == 'model_accuracy_KL':
            # Compute KL divergence between true and estimated models
            kl = 0
            for s in range(ag.n_states):
                for a in range(ag.n_actions_per_state[s]):
                    kl += np.sum(env.transition_matrix[s, a, :] * np.log(env.transition_matrix[s, a, :] / ag.T[s, a, :]))
            kl /= np.sum(ag.n_actions_per_state)
            return kl 
        
        elif self.measure_type == 'uniform_state_visitation':
            # Compute RMSE between state visitation frequencies and uniform distribution
            state_freq = ag.Cs / np.sum(ag.Cs)
            err = np.sum((state_freq - 1 / ag.n_states) ** 2) / ag.n_states
            return sqrt(err)
        
        elif self.measure_type == 'uniform_state_visitation_KL':
            # Compute KL divergence between state visitation frequencies and uniform distribution
            state_freq = ag.Cs / np.sum(ag.Cs)
            kl = np.sum(state_freq * np.log(state_freq / (1 / ag.n_states)))
            return kl
               
    
    def measure_performance(self, ag, env):
        """
        Run the agent in the environment for n_iter steps and evaluate performance regularly.
        Args:
            ag: Agent object
            env: Environment object
        Returns:
            perfs: Array of shape (n_repet, n_iter // eval_every) containing performance values
        """

        perfs = np.zeros(self.n_iter // self.eval_every)
            
        for t in range(self.n_iter//self.eval_every):
            run_steps(ag, env, self.eval_every)
            performance = self.evaluate(ag, env)
            perfs[t] = performance
            if self.verbose:
                print(f"Performance at iteration, {(t+1)*self.eval_every}: {performance}")
        return perfs