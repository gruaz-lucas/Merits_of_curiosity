import random
from src import RoomEnvironment


# ------------------------------------------------------------------------------
# The script below demonstrates how to render the environment
# ------------------------------------------------------------------------------

if __name__ == "__main__":
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

    # Reset the environment to start a new episode
    obs, info = env.reset()
    print(f"Initial state: {obs['state']}")
    done = False
    step_count = 0
    max_steps = 50  # Maximum number of steps to run

    while not done and step_count < max_steps:
        # Sample a random valid action
        valid_actions = obs['available_actions']
        if len(valid_actions) == 0:
            print(f"No valid actions from state {env.state}.")
            break
        action = random.choice(valid_actions)

        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step_count}: Action {action}, State {obs['state']}, Reward {reward}")

        # Optionally render the environment
        env.render()
        step_count += 1

    # Close the environment at the end
    env.render()
    env.close()