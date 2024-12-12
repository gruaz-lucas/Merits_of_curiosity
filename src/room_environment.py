import numpy as np
import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def randProb(p):
    """
    Returns True with probability p.
    """
    return p > np.random.rand()

def pop_random(v):
    """
    Removes and returns a random element from the list v.
    """
    index = random.randrange(len(v))
    return v.pop(index)

class Grid:
    """
    Class representing a grid (room) in the environment.
    """
    def __init__(self, node_ids, boundaries, is_sink=False, is_source=False, is_stoc=False):
        """
        Initializes a Grid object.

        Parameters:
        - node_ids: List of node IDs in the grid.
        - boundaries: Boundaries of the grid.
        - is_sink: Boolean indicating if the grid is a sink.
        - is_source: Boolean indicating if the grid is a source.
        - is_stoc: Boolean indicating if the grid is stochastic.
        """
        self.node_ids = node_ids
        self.boundaries = boundaries
        self.is_sink = is_sink
        self.is_source = is_source
        self.is_stoc = is_stoc

class RoomEnvironment(gym.Env):
    """
    Custom Gym environment representing a maze with grid-like rooms.
    """
    def __init__(
        self,
        n_init_states,
        branching_rate,
        room_size,
        p_room,
        p_sink,
        p_source,
        p_stoc,
        n_edges_per_sink,
        n_edges_per_source,
        uncontrollability,
        seed = None
    ):
        """
        Initializes the RoomEnvironment.

        Parameters:
        - n_init_states: Number of states in the initial maze (without rooms).
        - branching_rate: Rate at which the maze branches (probability of creating a new intersection when adding a state).
        - room_size: Size of the border each room (rooms are squares of shape room_size x room_size).
        - p_room: Probability of a room being created.
        - p_sink: Probability of a room being a sink.
        - p_source: Probability of a room being a source.
        - p_stoc: Probability of a room being stochastic.
        - n_edges_per_sink: Number of edges added per sink room.
        - n_edges_per_source: Number of edges added per source room.
        - uncontrollability: Inside stochastic rooms, probability of actions to lead to a random neighbor instead of the intended destination.
        """
        super().__init__()
        self.n_init_states = n_init_states
        self.branching_rate = branching_rate
        self.room_size = room_size
        self.p_room = p_room
        self.p_sink = p_sink
        self.p_source = p_source
        self.p_stoc = p_stoc
        self.n_edges_per_sink = n_edges_per_sink
        self.n_edges_per_source = n_edges_per_source
        self.uncontrollability = uncontrollability
        self.grids = []
        self.g = None # Graph

        # Generate the graph and related properties
        self.x_coords, self.y_coords, self.sink_edges, self.source_edges = self.generate_graph()
        assert self.n_tot_states == len(self.g.nodes())
        self.n_actions_per_state = [self.g.out_degree(s) for s in range(self.n_tot_states)]
        self.max_actions = max(self.n_actions_per_state)
        self.transition_matrix, self.reward_matrix = self.generate_transition_matrix()
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Discrete(self.n_tot_states)

        # Plotting information
        self.color_palette = plt.cm.Accent.colors
        self.room_colors = {
            "sink":self.color_palette[0],
            "source":self.color_palette[1],
            "stoc":self.color_palette[2],
            "neutral":self.color_palette[3],
            "corridor":self.color_palette[4],
        }
        self.fig, self.ax, self.img = None, None, None

        # Initialize the environment
        self.reset(seed=seed)

    def reset(self, seed=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed)
        self.state = np.random.randint(self.n_tot_states)
        obs = {
            'state': self.state,
            'available_actions': np.arange(self.n_actions_per_state[self.state])
        }
        return obs, {}

    def step(self, action):
        """
        Take an action and return the result.

        Parameters:
        - action: The action to take.

        Returns:
        - obs: The observation (state) after the action.
        - reward: The extrinsic reward obtained by taking the action (= 0 in our experiments).
        - terminated: Whether the episode has ended (False in our experiments).
        - truncated: Whether the episode was truncated (False in our experiments).
        - info: Additional information (None in our experiments).
        """
        if action >= self.n_actions_per_state[self.state]:
            raise ValueError(f"Invalid action {action} for state {self.state}.")

        # Get the transition probabilities for the current state and action
        probabilities = self.transition_matrix[self.state, action, :]
        next_state = np.random.choice(np.arange(self.n_tot_states), p=probabilities)

        # Get the reward
        reward = self.reward_matrix[self.state, action, next_state]

        # Update the state
        self.state = next_state
        terminated, truncated, info = False, False, None

        # Observations
        obs = {'state': self.state, 'available_actions': np.arange(self.n_actions_per_state[self.state])}

        # Return (observation, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        # For now, we'll display the graph with the current state highlighted
        pos = {s: (self.x_coords[s], self.y_coords[s]) for s in range(self.n_tot_states)}
        node_rooms = self.compute_node_properties()
        node_colors = [self.room_colors[node_rooms[i]] for i in range(self.n_tot_states)]
        node_sizes = [700/np.sqrt(self.n_tot_states)] * self.n_tot_states

        # Highlight the current state
        node_colors[self.state] = 'red'
        node_sizes[self.state] *= 2

        # Prepare edge colors
        base_colors = ['gray'] * self.g.number_of_edges()
        edge_list = list(self.g.edges())
        edge_width = 7.0/np.sqrt(self.n_tot_states)

        # Assign colors to sink edges
        sink_edge_indices = [edge_list.index(e) for e in self.sink_edges if e in edge_list]
        for idx in sink_edge_indices:
            base_colors[idx] = self.room_colors["sink"]

        # Assign colors to source edges
        source_edge_indices = [edge_list.index(e) for e in self.source_edges if e in edge_list]
        for idx in source_edge_indices:
            base_colors[idx] = self.room_colors["source"]

        # Create labels for nodes
        labels = {s: s for s in range(self.n_tot_states)}
        font_size = 50.0/np.sqrt(self.n_tot_states)

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        
        # Draw graph with node and edge attributes
        nx.draw(
            self.g, pos, ax=self.ax, node_color=node_colors, 
            node_size=node_sizes, edge_color=base_colors, labels=labels, 
            with_labels=True, font_size=font_size, arrows=True, width=edge_width
        )
        nx.draw_networkx_nodes(
            self.g, pos, ax=self.ax, node_color=node_colors, 
            node_size=node_sizes, edgecolors='black', linewidths=edge_width/2
        )

        # Pause to update the rendering
        plt.pause(0.01)

    def close(self):
        """
        Close the environment rendering.
        """
        if self.fig:
            plt.close(self.fig)

    def n_neighbors(self, s):
        """
        Return the number of neighbors of s in the graph. 
        s can be a single state or a room (Grid).
        """
        if isinstance(s, Grid):
            return 4 - len(s.boundaries)
        else:
            return len(list(self.g.successors(s)))

    def add_grid(self, previous_node, start_node, is_sink=False, is_source=False, is_stoc=False):
        """
        Create a grid room in the graph g, connected to previous_node, starting from start_node (first node id in the grid is start_node).
        """
        n = self.room_size
        node_ids = []
        boundaries = []

        # Iterate over rows and columns to create nodes and connect them
        for i in range(n):
            for j in range(n):
                # i is row, j is column
                state = start_node + i * n + j
                t = start_node + (i - 1) * n + j  # Top state
                d = start_node + (i + 1) * n + j  # Down state
                r = state + 1                     # Right state
                l = state - 1                     # Left state

                # Connect up, right, down, left directions if within bounds
                if i != 0:
                    self.g.add_edge(state, t)
                if j != n - 1:
                    self.g.add_edge(state, r)
                if i != n - 1:
                    self.g.add_edge(state, d)
                if j != 0:
                    self.g.add_edge(state, l)

                node_ids.append(state)

                # Connect to the previous node at the middle of the top boundary
                if i == 0 and j == (n - 1) // 2 and previous_node is not None:
                    self.g.add_edge(state, previous_node)
                    self.g.add_edge(previous_node, state)

                # Track boundaries
                if (i == n - 1 and j == (n - 1) // 2) or \
                   (previous_node is None and i == 0 and j == (n - 1) // 2) or \
                   ((j == 0 or j == n - 1) and i == (n - 1) // 2):
                    boundaries.append(state)

        # Create a new Grid object and append it to grids
        grid = Grid(node_ids, boundaries, is_sink, is_source, is_stoc)
        self.grids.append(grid)

        # Return the grid and the id of the last state in the grid
        last_state = start_node + (n - 1) * (n + 1)
        return grid, last_state

    def generateBaseStructure(self):
        """
        Create the graph structure with branches and grid rooms.
        No grid properties are assigned at this stage.
        """
        # Calculate the number of rooms to create
        n_rooms = int(self.n_init_states * self.p_room)
        room_ids = random.sample(range(self.n_init_states), n_rooms)  # IDs of nodes that will be transformed into rooms
        self.n_tot_states = self.n_init_states + n_rooms * (self.room_size ** 2 - 1)
        self.grids = []
        self.g = nx.DiGraph()
        self.g.add_nodes_from(range(self.n_tot_states))
        D = deque()
        D.append(None)
        next_node = 0  # Next node id to be used

        for i in range(self.n_init_states):
            s = D.popleft()
            
            while isinstance(s, Grid) and len(s.boundaries) == 0:
                # Drop grid with no boundaries
                if len(D) == 0:
                    s = None
                    break
                s = D.popleft()

            # With some probability, put it back in the queue (if not already full neighbors)
            if randProb(self.branching_rate) and s is not None and self.n_neighbors(s) < 3:
                D.appendleft(s)

            if isinstance(s, Grid):
                if len(s.boundaries) == 0:
                    continue
                cur_node = pop_random(s.boundaries)
            else:
                cur_node = s

            if i in room_ids:
                # Add a grid room
                grid, last_state = self.add_grid(cur_node, next_node)
                D.append(grid)
                next_node = last_state + 1
            else:
                # Connect to a new node if not part of a room
                if s is not None:
                    self.g.add_edge(cur_node, next_node)
                    self.g.add_edge(next_node, cur_node)
                D.append(next_node)
                next_node += 1
        return

    def assign_grid_properties(self):
        """
        Assign properties (sink, source, stochastic) to each grid in the environment.
        """
        n_rooms = len(self.grids)
        n_sink = int(n_rooms * self.p_sink)  # Number of sink rooms
        n_source = int(n_rooms * self.p_source)  # Number of source rooms
        n_stoc = int(n_rooms * self.p_stoc)  # Number of stochastic rooms
        all_indices = set(range(n_rooms))
        sink_ids = random.sample(list(range(n_rooms)), n_sink)
        remaining = all_indices - set(sink_ids)
        source_ids = random.sample(list(remaining), n_source)
        remaining -= set(source_ids)
        stoc_ids = random.sample(list(remaining), n_stoc)

        for i in sink_ids:
            self.grids[i].is_sink = True
        for i in source_ids:
            self.grids[i].is_source = True
        for i in stoc_ids:
            self.grids[i].is_stoc = True
        return

    def add_random_edges(self, from_nodes, to_nodes, n_e, max_tries = 10000):
        """
        Add random edges from a given set of nodes to another set of nodes.
        """
        n_added = 0
        added_edges = []
        i = 0

        # Add edges until the required number is reached
        while n_added < n_e and i < max_tries:
            sour = random.choice(from_nodes)
            dest = random.choice(to_nodes)

            # Check if the edge already exists
            if not self.g.has_edge(sour, dest):
                self.g.add_edge(sour, dest)
                added_edges.append((sour, dest))
                n_added += 1
            i += 1
        if i >= max_tries:
            raise ValueError(f"Could not add {n_e} edges after {max_tries} tries.")
        return added_edges

    def add_sink_source(self):
        """
        Add sink and source edges to the graph.
        """
        sink_edges = []
        source_edges = []
        all_nodes = set(self.g.nodes())

        for grid in self.grids:
            if grid.is_sink:
                from_nodes = list(all_nodes - set(grid.node_ids))
                to_nodes = grid.node_ids
                edges = self.add_random_edges(from_nodes, to_nodes, self.n_edges_per_sink)
                sink_edges.extend(edges)
            if grid.is_source:
                from_nodes = grid.node_ids
                to_nodes = list(all_nodes - set(grid.node_ids))
                edges = self.add_random_edges(from_nodes, to_nodes, self.n_edges_per_source)
                source_edges.extend(edges)

        return sink_edges, source_edges

    def compute_stoc_v(self):
        """
        Return the stochasticity of the nodes.

        stoc_v: stochasticity value for each node (True or False)
        """
        self.stoc_v = np.zeros(self.n_tot_states, dtype=bool)
        for grid in self.grids:
            if grid.is_stoc:
                for n in grid.node_ids:
                    self.stoc_v[n] = True
        return self.stoc_v

    def generate_graph(self):
        """
        Generate the whole graph with rooms and room properties.
        """
        self.generateBaseStructure()
        self.assign_grid_properties()

        # Generate positions for plotting (before adding sink/source edges)
        pos = nx.kamada_kawai_layout(self.g)
        x_coords = [pos[node][0] for node in self.g.nodes()]
        y_coords = [pos[node][1] for node in self.g.nodes()]

        # Add sink and source edges
        sink_edges, source_edges = self.add_sink_source()

        # Compute stochastic nodes
        self.compute_stoc_v()

        return x_coords, y_coords, sink_edges, source_edges

    def generate_transition_matrix(self):
        """
        Generate transition and reward matrices.
        """
        Psa_s = np.zeros((self.n_tot_states, self.max_actions, self.n_tot_states))
        Rsa_s = np.zeros((self.n_tot_states, self.max_actions, self.n_tot_states))

        for u in range(self.n_tot_states):
            # Get neighbors and their indices for the current state
            neighbors = list(self.g.successors(u))

            for ia, v in enumerate(neighbors):
                if self.stoc_v[u]:
                    # Set transition probabilities for stochastic nodes
                    prob = self.uncontrollability / len(neighbors)
                    Psa_s[u, ia, neighbors] = prob
                    Psa_s[u, ia, v] += 1.0 - self.uncontrollability
                else:
                    # Set transition probabilities for deterministic nodes
                    Psa_s[u, ia, v] = 1.0

                # Rewards are zero
                Rsa_s[u, ia, v] = 0.0

        return Psa_s, Rsa_s

    def compute_node_properties(self):
        """
        Compute properties of nodes in the graph.
        """
        props = ["corridor" for i in range(self.n_tot_states)]  # Default corridor
        for grid in self.grids:
            grid_prop = "neutral"
            if grid.is_stoc:
                grid_prop = "stoc"
            elif grid.is_sink:
                grid_prop = "sink"
            elif grid.is_source:
                grid_prop = "source"
            for n in grid.node_ids:
                props[n] = grid_prop
        return props
