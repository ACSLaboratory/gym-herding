"""
Graph Class.

Using this class, a grid node-graph may be constructed
for use in herding scenarios.

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import Tuple
from typing import Optional
from typing import Dict

import numpy as np

from gym_herding.envs.graph.node import Node
from gym_herding.envs.graph.distribution import Distribution
from gym_herding.envs.utils.position import to_matrix


class NodeGraph():
    """
    Node graph containing nodes.

    Parameters
    ----------
    n_v : int
        The number of vertices that must be squared.

    Attributes
    ----------
    node : Node()
        Something. Cannot be overwritten.

    Methods
    -------

    """

    def __init__(self, n_v: int, n_p: int, weights: List[float]) -> None:
        """Initialize the NodeGraph class."""
        self._node_dict: Dict[int, Node] = {}
        self.distribution = Distribution(n_v, n_p, weights)

        # Instantiate n_v^2 number of nodes and thier initial positions
        state_id = 0
        for i in range(0, n_v):
            for j in range(0, n_v):
                self._node_dict[state_id] = Node(state_id, i, j)
                state_id += 1

        # Immutable parameter variables
        self._param = {
            'n_v': n_v,
            'n_p': n_p,
            'weights': weights,
            'size_fraction': 1 / n_p,
            'total_states': np.power(n_v, 2),
            'iter_count': 0,
            'all_positions': [[i, j] for i in range(n_v) for j in range(n_v)],
        }

    @property
    def node(self):
        """Returns the node dictionary."""
        return self._node_dict

    def get_position(self, state: int) -> Optional[np.ndarray]:
        """Given a state, return the node's xy position."""
        if state in self._node_dict:
            return self._node_dict[state].position

        return None

    def get_state(self, pos) -> Optional[int]:
        """Given an xy position, return the nodes state id."""
        for node in self._node_dict.values():
            if all(pos == node.position):
                return node.state_id

        return None

    def set_node_neighbors(self) -> None:
        """Set the neighbors for each node."""
        state = 0
        for i in range(0, self._param['n_v']):
            for j in range(0, self._param['n_v']):
                # Craete temporary neighbor storage variable and fill
                temp_neigh = []

                if i + 1 < self._param['n_v']:
                    temp_neigh.append(np.array([i + 1, j]))

                if j - 1 >= 0:
                    temp_neigh.append(np.array([i, j - 1]))

                if i - 1 >= 0:
                    temp_neigh.append(np.array([i - 1, j]))

                if j + 1 < self._param['n_v']:
                    temp_neigh.append(np.array([i, j + 1]))

                # Set neighbors for the specific node.
                self._node_dict[state].neighbors = np.asarray(temp_neigh)
                state += 1

    def set_node_positions(self) -> None:
        """
        Sets individual node XY positions.

        For each node in the graph, an xy position is given in a plotting
        format. This means that a coordinate (0, 0) means the bottom left
        of a cartesian xy-plot. To convert to an ij-position scheme, use
        the `to_matrix()` function in the `position.py` file.

        TODO: Update docs bc now we keep Matrix form.
        """
        state = 0
        for i in range(0, self._param['n_v']):
            for j in range(0, self._param['n_v']):
                # Set position for the specific node.
                self._node_dict[state].position = np.array(
                    [i, j], dtype=np.int8)
                state += 1

    def set_node_jump_rates(self, beta: float) -> None:
        """Sets individual node jump rates (beta value)."""
        for i in range(0, self._param['total_states']):
            self._node_dict[i].beta = beta

    def convert_action_to_node_info(
            self,
            old_state: int,
            action: int) -> Tuple[int, int, int, bool]:
        """
        Converts a leader movement action to Graph properties.

        From a current state, the leader agent's motion to a new state is
        reflected by the action it pursues. This function returns the
        xy-coordinate position and state ID of the node that the leader
        has moved to.

        Parameters
        ----------
        old_state : int
            The current state of the leader agent.
        action : {0, 1, 2, 3, 4}, int
            Possible actions that a leader agent undergoes. Each int option
            represents Left, Right, Up, Down, Stay, respectively.

        Returns
        -------
        i, j : int
            The coordinates of the new node that the leader agent moved to.
        state_id : int
            The state id of the new node that the leader agent moved to.
        is_out_of_bounds : bool
            If the action puts the leader into an out of bounds node, this
            variable will return True.

        Note
        ----
        If the leader goes out of bounds, the agent remains in the original
        state.

        """
        # Get new state from action
        state_id: bool = None
        is_out_of_bounds: bool = False
        if action == 0:  # Left
            state_id, is_out_of_bounds = self.action_left(old_state)

        elif action == 1:  # Right
            state_id, is_out_of_bounds = self.action_right(old_state)

        elif action == 2:  # Up
            state_id, is_out_of_bounds = self.action_up(old_state)

        elif action == 3:  # Down
            state_id, is_out_of_bounds = self.action_down(old_state)

        elif action == 4:  # Stay
            state_id, is_out_of_bounds = self.action_stay(old_state)

        # Get new position from action
        i, j = self._node_dict[state_id].position

        return (i, j, state_id, is_out_of_bounds)

    def action_left(self, old_state: int) -> Tuple[int, bool]:
        """Node position change and boundary bool for leader moving left."""
        if old_state % self._param['n_v'] != 0:
            return (old_state - 1, False)

        return (old_state, True)

    def action_right(self, old_state: int) -> Tuple[int, bool]:
        """Node position change and boundary bool for leader moving right."""
        if (old_state  + 1) % self._param['n_v'] != 0:
            return (old_state + 1, False)

        return (old_state, True)

    def action_up(self, old_state: int) -> Tuple[int, bool]:
        """Node position change and boundary bool for leader moving up."""
        if old_state > (self._param['n_v'] - 1):
            return (old_state - self._param['n_v'], False)

        return (old_state, True)

    def action_down(self, old_state: int) -> Tuple[int, bool]:
        """Node position change and boundary bool for leader moving down."""
        if old_state < (self._param['n_v'] - 1):
            return (old_state + self._param['n_v'], False)

        return (old_state, True)

    def action_stay(self, old_state: int) -> Tuple[int, bool]:
        """Node position change and boundary bool for leader staying."""
        return (old_state, True)

    def update_count(self) -> None:
        """Updates the agent count in each node/vertix."""
        for pos in self._param['all_positions']:
            state = self.get_state(pos)
            mat_i, mat_j = to_matrix(self._param['n_v'], pos)
            self._node_dict[state].agent_count = \
                int(self.distribution.current[mat_i, mat_j] / \
                    self._param['size_fraction'])

    def reset(self) -> None:
        """Reset non-immutable values in all the Nodes."""
        self.distribution.reset()
        for i in range(0, np.power(self._param['n_v'], 2)):
            self._node_dict[i].reset()

    def __iter__(self):
        """Iterator for nodes/vertices of the graph."""
        self._param['iter_count'] = -1
        return self

    def __next__(self):
        """Iterate through the nodes/vertices of the graph."""
        # - 1 because zero index
        if self._param['iter_count'] < self._param['total_states'] - 1:
            self._param['iter_count'] += 1

        else:
            raise StopIteration

        return self._node_dict[self._param['iter_count']]

    def __repr__(self):
        """Returns offical representation of graph."""
        graph_representation = np.zeros(
            (self._param['n_v'], self._param['n_v']), dtype=np.int8)
        for node in self:
            mat_i, mat_j = to_matrix(self._param['n_v'], node.position)
            graph_representation[mat_i, mat_j] = node.state_id

        return np.array2string(graph_representation)
