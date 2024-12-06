import random
import numpy as np
from src import RoomEnvironment, NoveltyAgent, SurpriseAgent, RandomAgent, InformationGainAgent, EmpowermentAgent, Performance


# ------------------------------------------------------------------------------
# The script below to run an agent in the environment and compute performance
# ------------------------------------------------------------------------------

seed = None

if __name__ == "__main__":
    
    random.seed(seed)
    np.random.seed(seed)
    # Instantiate the environment with specified parameters
    env = RoomEnvironment(
        n_init_states=13,
        branching_rate=0.5,
        room_size=3,
        p_room=4/13,
        p_sink=0.25,
        p_source=0.25,
        p_stoc=0.25,
        n_edges_per_sink=10,
        n_edges_per_source=10,
        uncontrollability=1.0
    )
    
    # Instantiate the agent with specified parameters
    agent = InformationGainAgent(
        n_states=env.n_tot_states,
        n_actions_per_state=env.n_actions_per_state,
        ε=1/env.n_tot_states,
        βi=100.0,
        λi=0.9,
        βe=0.0,
        λe=0.0,
        T_PS=50,
        model_fixed=False,
        r0=0.0
    )
    
    # Instantiate the performance measure
    performance = Performance(
        measure_type='state_discovery',
        n_iter=1000,
        eval_every=50,
        verbose=True
    )

    # Reset the environment to start a new episode
    obs, info = env.reset(seed = seed)
    print(f"Initial state: {obs['state']}")

    # Run the agent in the environment and measure performance
    perfs = performance.measure_performance(agent, env)
    
    print(f"Average performance: {np.mean(perfs)}")
    
    
    