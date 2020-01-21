"""
Node Class

A fundamental part of the grid-graph. Each vertex within the graph is comprised
of these Nodes.

Written by: Zahi Kakish (zmk5)

"""
from typing import Union
from typing import List
from typing import Any
from typing import Dict
import numpy as np


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
            "jump_weight": 0.1,
        }
        self.num_of_neighbors = 0
        self.max_neighbors = max_neighbors
        self.beta = 0.1  # Jump Weight

    @property
    def beta(self) -> float:
        """ Beta value getter (Jumping Weight) """
        return self._param["jump_weight"]

    @beta.setter
    def beta(self, new_beta: float) -> None:
        """ Beta value setter (Jumping Weight) """
        self._param["jump_weight"] = new_beta

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
