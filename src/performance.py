
import numpy as np
from .room_environment import RoomEnvironment

def run_steps(ag, env, n_steps):
    for _ in range(n_steps):
        state = env.state
        a = ag.sample_action(state)
        obs, reward, done, truncated, info = env.step(a)
        ag.update_model(state, a, obs['state'], reward)
    return ag, env


class Performance():
    """
    Class to measure the performance of an agent during simulation.
    """
    
    def __init__(self, measure_type, n_iter, eval_every, verbose=False):
        self.measure_type = measure_type
        self.n_iter = n_iter
        self.eval_every = eval_every
        self.verbose = verbose
        
    def evaluate(self, ag, env):
        if self.measure_type == 'state_discovery':
            unvisited = np.sum(ag.Cs == 0)
            return unvisited / ag.n_states
        
        elif self.measure_type == 'model_accuracy':
            # Compute MSE between true and estimated models
            err = 0
            for s in range(ag.n_states):
                for a in range(ag.n_actions_per_state[s]):
                    err += np.sum((ag.T[s, a] - env.transition_matrix[s, a]) ** 2)
            err /= ag.n_states * np.sum(ag.n_actions_per_state)
            return err
        
        elif self.measure_type == 'uniform_state_visitation':
            # Compute MSE between state visitation frequencies and uniform distribution
            state_freq = ag.Cs / np.sum(ag.Cs)
            err = np.sum((state_freq - 1 / ag.n_states) ** 2) / ag.n_states
            return err
               
    
    def measure_performance(self, ag, env):
        perfs = []
        for t in range(0, self.n_iter, self.eval_every):
            run_steps(ag, env, self.eval_every)
            performance = self.evaluate(ag, env)
            perfs.append(performance)
            if self.verbose:
                print(f"Performance at iteration {t}: {performance}")
        return perfs
