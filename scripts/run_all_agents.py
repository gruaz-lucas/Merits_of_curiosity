import random
import numpy as np
from src import RoomEnvironment, NoveltyAgent, SurpriseAgent, RandomAgent, InformationGainAgent, EmpowermentAgent, SPIEAgent, MOPAgent, Simulator
import time


# -------------------------------------------------------------
# Run multiple agents in environments and save performances
# -------------------------------------------------------------

if __name__ == "__main__":
    
    # Simulation parameters
    seed = None
    n_envs = 10
    n_iter = 500
    eval_every = 20
    measure = 'model_accuracy'
    model_fixed = False
    data_path = 'data/'
    filename = 'source_m2_all_agents.npz'
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Agents to evaluate
    agent_names = ['Novelty', 'Surprise', 'InformationGain', 'Empowerment', 'MOP', 'SPIE', 'Random']
    agent_classes = [NoveltyAgent, SurpriseAgent, InformationGainAgent, EmpowermentAgent, MOPAgent, SPIEAgent, RandomAgent]
    
    # inverse temperature parameter for each agent (arbitrary here but can be optimized)
    βis = [50.0, 50.0, 50.0, 0.05, 0.0, 3.0, 0.0] 
    
    # Instantiate the simulator
    simulator = Simulator(
        measure_type=measure,
        n_iter=n_iter,
        eval_every=eval_every,
        verbose=False
    )
    
    # Instantiate the environment with specified parameters (source environment here)
    envs = []
    for e in range(n_envs):
        envs.append(RoomEnvironment(
            n_init_states=40,
            branching_rate=0.2,
            room_size=4,
            p_room=4/40,
            p_sink=0.0,
            p_source=0.25,
            p_stoc=0.0,
            n_edges_per_sink=50,
            n_edges_per_source=50,
            uncontrollability=1.0,
            seed = e
        ))
        
    # Initialize the performance matrix
    perfs = np.zeros((len(agent_classes), n_envs, n_iter // eval_every))
    
    # Run the agents in the environments
    for i, ag_class in enumerate(agent_classes):
        for j, env in enumerate(envs):
            
            print(f"Running agent {ag_class.__name__} in environment {j}")
            # Instantiate an agent of the specified class
            ag = ag_class(
                n_states=env.n_tot_states,
                n_actions_per_state=env.n_actions_per_state,
                ε=1/env.n_tot_states,
                βi=βis[i],
                λi=(0.5)**(2/env.n_tot_states),
                βe=0.0,
                λe=0.0,
                model_fixed=model_fixed
            )
            
            # If the agent has a fixed model, set the transition matrix to the environment's transition matrix
            if model_fixed:
                ag.model_fixed = True
                ag.T = env.transition_matrix.copy()
                
            t0 = time.time()
            # Run the agent in the environment
            perfs[i, j, :] = simulator.measure_performance(ag, env)
            print(f"Elapsed time: {time.time() - t0}")
    
    # Save the performance data
    np.savez(data_path + filename, perfs=perfs, agent_names=agent_names, envs=[env.__dict__ for env in envs], n_iter=n_iter, eval_every=eval_every, βis=βis)
    
    print(f"Performance saved under {data_path + filename}")
    
    
    