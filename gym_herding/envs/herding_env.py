"""
Herding Experiment environment:

Description

Action Space:
    - Where to move: (Up, Down, Left, Right, Stay)

Written by: Zahi Kakish (zmk5)

"""
from typing import NoReturn
from typing import Tuple
from typing import Optional

import numpy as np

from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

from gym_herding.envs.graph.graph import NodeGraph
from gym_herding.envs.graph.leader import Leader
from gym_herding.envs.utils.parameters import HerdingEnvParameters
from gym_herding.envs.plotting.agent_view import HerdingEnvPlotting
from gym_herding.envs.plotting.bar_view import HerdingEnvPlottingBar


class HerdingEnv(Env):
    """
    Herding OpenAI Environment.

    Parameters
    ----------
    hep : HerdingEnvParameters
        Parameters required for the Herding Environment.
    observation_space : {1, 2, 3}, optional
        Choose the size of the observation space that is returned by the
        `step()` method and used by the remainder of the `HerdingEnv`.

    """

    def __init__(self, hep: Optional[HerdingEnvParameters] = None,
                 observation_space: int = 3) -> None:
        """Initialize the HerdingEnv Class."""
        # Set immutable params and mutable variables, if applicable.
        self.param = hep
        self.graph: NodeGraph = None
        self.leader: Leader = None
        self.var = {
            'is_out_of_bounds': False,
        }

        if self.param is None:
            print('Please provide environment parameters in the ' +
                  'HerdingEnvParamters object\nbefore continuing!')

        else:
            if isinstance(hep, HerdingEnvParameters):
                self.initialize(hep, observation_space)

            else:
                raise TypeError('First argument must be a ' +
                                'HerdingEnvParameter object!')

    def initialize(self,
                   hep: HerdingEnvParameters,
                   observation_space: int = 3) -> None:
        """Initialize the OpenAI Environment."""
        # Set immutable params
        self.param = hep

        # Initialize Graph, Agent, and Leader values.
        self._initialize_graph()

        # Observation Space
        self._initialize_observation(observation_space)

        # Action Space
        self._initialize_actions()

        # Set plotting class
        if self.param.extra['rendering_enabled']:
            if self.param.extra['visualization'] == 'graph':
                self._plot = HerdingEnvPlotting(
                    self.param.n_v, self.param.n_p)

            else:
                self._plot = HerdingEnvPlottingBar(
                    self.param.n_v, self.param.n_p)

            # Initialize rendering environment (matplotlib)
            self._plot.create_figure()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        """Execute the given action."""
        # Get new leader state and position from action
        new_lx, new_ly, new_ls, is_out_of_bounds = \
            self.graph.convert_action_to_node_info(self.leader.state, action)

        # If action takes leader out of bounds, change info state to say that
        # occured.
        self.var['is_out_of_bounds'] = is_out_of_bounds

        # Apply changes to leaders position and state onto the class
        self._move_leader(new_ls, new_lx, new_ly)

        # Apply repulsive effect of leader in new location on those local
        # herding agents.
        if self.param.extra['leader_motion_moves_agents']:
            self._move_herding_agents()

        # Apply repulsive effect of leader in new location on those local
        # herding agents ONLY IF action STAY is used.
        else:
            if action == 4:  # Stay
                self._move_herding_agents()

        # Get newest observation of the state space.
        obs = self._get_observation()

        # Get reward for action
        reward = self._get_reward()

        # Write info
        info = {
            'is_out_of_bounds': self.var['is_out_of_bounds'],
            'leader_state': self.leader.state,
            'leader_pos': [new_lx, new_ly],
        }

        # Check for end of episode
        if self.param.iter == self.param.max_iter:
            done = True

        else:
            self.param.iter += 1
            done = False

        return (obs, reward, done, info)

    def reset(self) -> np.ndarray:
        """Initializies the environment to initial state for next episode."""
        self.graph.reset()
        self.graph.update_count()
        self.leader.reset()
        self.param.iter = 0
        self.param.extra['t'] = 0

        # Returns the current reset observational space.
        return self.graph.distribution.current

    def render(self, mode: str = 'human') -> None:
        """Display the environment."""
        if self.param.extra['rendering_enabled']:
            self._plot.render(self.graph, self.leader)

        else:
            print('Rendering was disabled for this environment!')

    def save_render(self, file_name: str) -> None:
        """Save image of the render."""
        if self.param.extra['rendering_enabled']:
            self._plot.save_render(file_name)

        else:
            print('Rendering was disabled for this environment!')

    def close(self) -> NoReturn:
        """Close the environment."""
        raise NotImplementedError()

    def is_action_valid(self, action: int) -> bool:
        """Check if the leader agent action is valid."""
        if action == 0:  # Left
            if self.leader.state % self.param.n_v != 0:
                return True

        elif action == 1:  # Right
            if (self.leader.state + 1) % self.param.n_v != 0:
                return True

        elif action == 2:  # Up
            if self.leader.state < (self.param.n_v * (self.param.n_v - 1)):
                return True

        elif action == 3:  # Down
            if self.leader.state > self.param.n_v:
                return True

        elif action == 4:  # Stay
            return True

        return False

    def _initialize_env_objects(self) -> None:
        """Initialize environment object values."""
        self.graph.distribution.target = self.param.dist['target']
        self.graph.distribution.initial = self.param.dist['initial']
        for dist in ['initial', 'current', 'target']:
            self.graph.distribution.apply_population(dist)

        self.graph.set_node_positions()
        self.graph.set_node_neighbors()
        self.graph.update_count()

    def _get_reward(self) -> NoReturn:
        """Get reward based on results of leader action."""
        raise NotImplementedError()

    def _get_observation(self) -> NoReturn:
        """Get the calculated observation of environment state."""
        raise NotImplementedError()

    def _move_herding_agents(self) -> None:
        """Move the herding agents within the environment."""
        # Get number of agents within the node that the leader is in
        chk_ld = self.graph.node[self.leader.state].agent_count
        beta = self.graph.node[self.leader.state].beta  # Jump Weight

        # Increment time
        self.param.extra['t'] += self.param.extra['dt']

        # Generate random number to see if agent should change node
        if chk_ld != 1:
            toss = np.squeeze(np.random.rand(1, chk_ld))

        else:
            # Squeeze reduces too much if chk_ld == 1
            toss = np.random.rand(1, chk_ld)[0]

        # Get available neighboring nodes from where the leader is located
        avail_neighbors = self.graph.node[self.leader.state].neighbors

        # Get number of agents that will jump and where they jump
        jumping_agents = len(np.where(toss <= len(avail_neighbors) * beta)[0])
        rand_perm = np.random.randint(
            0, len(avail_neighbors) + 1, jumping_agents)

        for i in range(len(avail_neighbors)):
            jump_count = len(np.where(rand_perm == i)[0])
            jump_state = self.graph.get_state(avail_neighbors[i, :])

            # Helpful variables
            old_x, old_y = self.graph.get_position(self.leader.state)
            new_x, new_y = self.graph.get_position(jump_state)

            # Update current distribution map
            self.graph.node[self.leader.state].agent_count -= jump_count
            self.graph.node[jump_state].agent_count += jump_count
            self.graph.distribution.increment_node_value(
                -1 * jump_count, old_x, old_y, 'current')
            self.graph.distribution.increment_node_value(
                jump_count, new_x, new_y, 'current')

    def _move_leader(self, new_ls: int, new_lx: int, new_ly: int) -> None:
        """Move the leader to the next position."""
        self.leader.state = new_ls
        self.leader.real = np.array([new_lx, new_ly], dtype=np.int8)
        self.leader.visual = np.array([new_lx, new_ly], dtype=np.int8)

    def _initialize_graph(self):
        """Instantiate Graph, Agents, and Leader object."""
        self.graph = NodeGraph(
            self.param.n_v, self.param.n_p, self.param.weights)
        self.leader = Leader(
            self.param.extra['init_leader_state'], self.param.n_v,
            self.param.extra['init_leader_pos'][0],
            self.param.extra['init_leader_pos'][1])

        # Set Node Jump Weight (Beta)
        self.graph.set_node_jump_rates(self.param.extra['jump_weight'])

        self.graph.distribution.target = self.param.dist['target']
        self.graph.distribution.initial = self.param.dist['initial']
        for dist in ['initial', 'current', 'target']:
            self.graph.distribution.apply_population(dist)

        self.graph.set_node_positions()
        self.graph.set_node_neighbors()
        self.graph.update_count()

    def _initialize_observation(self, observation_space: int):
        """Initialize the observation space."""
        # Testing with multiple observation spaces
        #     Test 1: Dist in the current node (1)
        #     Test 2: Dist in left, right, up, down, and current node (5)
        #     Test 3: The whole state-space (n_v^2)
        if observation_space == 1:
            self.observation_space = Box(
                low=0, high=1, shape=(1,), dtype=np.float32)

        elif observation_space == 2:
            self.observation_space = Box(
                low=0, high=1, shape=(5,), dtype=np.float32)

        elif observation_space == 3:
            self.observation_space = Box(
                low=0, high=1, shape=(self.param.n_v, self.param.n_v),
                dtype=np.float32)

    def _initialize_actions(self):
        """Initialize the action space."""
        # Should be only possible path actions that the agent can go through,
        # and in this case it is either:
        #     0 - Left
        #     1 - Right
        #     2 - Up
        #     3 - Down
        #     4 - Stay
        self.action_space = Discrete(5)
