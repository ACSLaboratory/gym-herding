"""
Graph and Node Classes

Using these classes, a grid node-graph may be constructed
for use in herding scenarios.

Written by: Zahi Kakish (zmk5)

"""
from typing import Union
from typing import List
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
import numpy as np
from gym_herding.envs.distribution import Distribution
from gym_herding.envs.position import to_matrix


class Node():
    """
    Node on a Graph

    Parameters
    ----------
    state_id : int
        The numerical ID of the node.
    pos_x : int
        Initial x-coordinate position.
    pos_y : int
        Initial y-coordinate position.
    max_neighbors : int, optional
        Maximum number of neighbor nodes. Defaults to 4 neighbors.

    Attributes
    ----------


    """
    def __init__(self, node_id: int, pos_x: int, pos_y: int,
                 max_neighbors: int = 4) -> None:
        self._node_id = node_id
        self._position = np.array([pos_x, pos_y], dtype=np.int8)
        self._neighbors = 0
        self._param = {
            "agent_count": 0,
            "init_pos_x": pos_x,
            "init_pos_y": pos_y,
        }
        self.num_of_neighbors = 0
        self.max_neighbors = max_neighbors

    @property
    def state_id(self) -> int:
        """ State ID getter property """
        return self._node_id

    @state_id.setter
    def state_id(self, node_id: int) -> None:
        """ State ID setter property """
        self._node_id = node_id

    @property
    def agent_count(self) -> int:
        """ Agent count getter property """
        return self._param["agent_count"]

    @agent_count.setter
    def agent_count(self, val: int) -> None:
        """ Agent count setter property """
        self._param["agent_count"] = val

    @property
    def position(self) -> np.ndarray:
        """ Position getter property """
        return self._position

    @position.setter
    def position(self,
                 pos_xy: Union[np.ndarray, Dict[int, int], List[int]]) -> None:
        """ Position setter property """
        if isinstance(pos_xy, np.ndarray):
            self._position = pos_xy

        elif isinstance(pos_xy, (dict, list)):
            self._position = np.array([pos_xy[0], pos_xy[1]], dtype=np.int8)

    @property
    def neighbors(self) -> np.ndarray:
        """ Neighbors getter property """
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors: np.ndarray) -> None:
        """ Neighbors setter property """
        if isinstance(neighbors, np.ndarray):
            if (neighbors.shape[0] <= self.max_neighbors and
                    neighbors.shape[1] <= 2):
                self._neighbors = neighbors

            else:
                raise IndexError(
                    "Neighbors should be a (%d x 2) numpy.ndarray." \
                    % self.max_neighbors)

        else:
            raise IndexError("Neighbors should be a (%d x 2) numpy.ndarray." \
                             % self.max_neighbors)

    def set_param(self, key: str, val: Any, new: bool = False) -> None:
        """ something """
        if new or key in self._param:
            self._param[key] = val

        else:
            raise KeyError(str(key) + " is not a valid parameter.")

    def reset(self) -> None:
        """ Reset non-immutable values of Node """
        self._param["agent_count"] = 0


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
        self._node_dict = {}
        self.distribution = Distribution(n_v, n_p, weights)

        # Instantiate n_v^2 number of nodes
        for i in range(0, np.power(n_v, 2)):
            self._node_dict[i] = Node(i, 0, 0)

        # Immutable parameter variables
        self._param = {
            "n_v": n_v,
            "n_p": n_p,
            "weights": weights,
            "size_fraction": 1 / n_p,
            "total_states": np.power(n_v, 2),
            "iter_count": 0,
            "all_positions": [[i, j] for i in range(n_v) for j in range(n_v)],
        }

    @property
    def node(self):
        """ Returns the node dictionary """
        return self._node_dict

    def get_position(self, state: int) -> Optional[np.ndarray]:
        """ Given a state, return the node's xy position """
        if state in self._node_dict:
            return self._node_dict[state].position

        return None

    def get_state(self, pos) -> Optional[int]:
        """ Given an xy position, return the nodes state id """
        for node in self._node_dict.values():
            if all(pos == node.position):
                return node.state_id

        return None

    def set_node_neighbors(self) -> None:
        """ Set the neighbors for each node. """
        state = 0
        for j in range(0, self._param["n_v"]):
            for i in range(0, self._param["n_v"]):
                # Craete temporary neighbor storage variable and fill
                temp_neigh = []

                if i + 1 < self._param["n_v"]:
                    temp_neigh.append(np.array([i + 1, j]))

                if j - 1 >= 0:
                    temp_neigh.append(np.array([i, j - 1]))

                if i - 1 >= 0:
                    temp_neigh.append(np.array([i - 1, j]))

                if j + 1 < self._param["n_v"]:
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

        """
        state = 0
        for j in range(0, self._param["n_v"]):
            for i in range(0, self._param["n_v"]):
                # Set position for the specific node.
                self._node_dict[state].position = np.array(
                    [i, j], dtype=np.int8)
                state += 1

    def convert_action_to_node_info(self, old_state: int,
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
        state_id = None
        is_out_of_bounds = False
        if action == 0:  # Left
            state_id, is_out_of_bounds = self.action_left(old_state)

        elif action == 1:  # Right
            state_id, is_out_of_bounds = self.action_right(old_state)

        elif action == 2:  # Up
            state_id, is_out_of_bounds = self.action_up(old_state)

        elif action == 3:  # Down
            state_id, is_out_of_bounds = self.action_down(old_state)

        elif action == 4:  # Stay
            state_id, is_out_of_bounds = self.action_down(old_state)

        # Get new position from action
        i, j = self._node_dict[state_id].position

        return (i, j, state_id, is_out_of_bounds)

    def action_left(self, old_state: int) -> Tuple[int, bool]:
        """ Node position change and boundary bool for leader moving left """
        if old_state % self._param["n_v"] != 0:
            return (old_state - 1, False)

        return (old_state, True)

    def action_right(self, old_state: int) -> Tuple[int, bool]:
        """ Node position change and boundary bool for leader moving right """
        if (old_state  + 1) % self._param["n_v"] != 0:
            return (old_state + 1, False)

        return (old_state, True)

    def action_up(self, old_state: int) -> Tuple[int, bool]:
        """ Node position change and boundary bool for leader moving up """
        if old_state < (self._param["n_v"] * (self._param["n_v"] - 1)):
            return (old_state + self._param["n_v"], False)

        return (old_state, True)

    def action_down(self, old_state: int) -> Tuple[int, bool]:
        """ Node position change and boundary bool for leader moving down """
        if old_state > (self._param["n_v"] - 1):
            return (old_state - self._param["n_v"], False)

        return (old_state, True)

    def action_stay(self, old_state: int) -> Tuple[int, bool]:
        """ Node position change and boundary bool for leader staying """
        return (old_state, True)

    def update_count(self) -> None:
        """ Updates the agent count in each node/vertix """
        for pos in self._param["all_positions"]:
            state = self.get_state(pos)
            mat_i, mat_j = to_matrix(self._param["n_v"], pos)
            self._node_dict[state].agent_count = \
                int(self.distribution.current[mat_i, mat_j] / \
                    self._param["size_fraction"])

    def reset(self) -> None:
        """ Reset non-immutable values in all the Nodes """
        self.distribution.reset()
        for i in range(0, np.power(self._param["n_v"], 2)):
            self._node_dict[i].reset()

    def __iter__(self):
        """ Iterator for nodes/vertices of the graph """
        self._param["iter_count"] = -1
        return self

    def __next__(self):
        """ Iterate through the nodes/vertices of the graph """
        # - 1 because zero index
        if self._param["iter_count"] < self._param["total_states"] - 1:
            self._param["iter_count"] += 1

        else:
            raise StopIteration

        return self._node_dict[self._param["iter_count"]]

    def __repr__(self):
        """ Returns offical representation of graph """
        graph_representation = np.zeros(
            (self._param["n_v"], self._param["n_v"]), dtype=np.int8)
        for node in self:
            mat_i, mat_j = to_matrix(self._param["n_v"], node.position)
            graph_representation[mat_i, mat_j] = node.state_id

        return np.array2string(graph_representation)
