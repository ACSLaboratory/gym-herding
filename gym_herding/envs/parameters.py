"""
Herding Environment Parameter Class

Written by: Zahi Kakish (zmk5)
"""
from typing import Any
from typing import Union
from typing import List
import numpy as np


class HerdingEnvParameters():
    """
    Parameters for use in the OpenAI Herding Environment.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The number of agents to herd
    weights : list
        The weight applied to the population distribution within the state
        space.
    beta : float
        Jump weight used by control matrix B in single linear system ODE
        representing a Kolmogorov forward equation.

    """
    def __init__(self, n_v: int, n_p: int,
                 weights: Union[List[float], np.ndarray],
                 beta: float = 0.1) -> None:
        self.iter = 0
        self.max_iter = 10000
        self.n_v = n_v
        self.n_p = n_p

        # Set weights
        if isinstance(weights, (list, np.ndarray)):
            if len(weights) != 2:
                raise ValueError("Can only be two weight values.")

            self.weights = weights

        else:
            self.weights = [1/10, 9/10]

        self.dist = {
            "target": 0,
            "initial": 0,
        }

        self.extra = {
            "init_leader_state": 0,
            "init_leader_pos": np.zeros(2, dtype=np.int8),
            "t": 0,
            "dt": 0.1,
            "jump_weight": beta,
            "leader_motion_moves_agents": False,
        }


    @property
    def beta(self) -> float:
        """ Beta value getter (Jumping Weight) """
        return self.extra["jump_weight"]

    @beta.setter
    def beta(self, new_beta: float) -> None:
        """ Beta value setter (Jumping Weight) """
        self.extra["jump_weight"] = new_beta

    def set_agents_distribution(self, val: np.ndarray, dist: str) -> None:
        """ Set the herding agents distribution in space. """
        if isinstance(val, np.ndarray):
            if val.shape != (self.n_v, self.n_v):
                raise ValueError("Distribution must be n_v x n_v")

            # Set distribution as value
            self.dist[dist] = val

        else:
            raise TypeError("Distribution must be a np.ndarray.")

    def set_leader_position(self, i: int, j: int) -> None:
        """ Set an initial postion for the leader. """
        self.extra["init_leader_postion"][0] = i
        self.extra["init_leader_position"][1] = j

    def add_extra_parameter(self, key: str, val: Any) -> None:
        """ Add an extra parameter, if needed."""
        self.extra[key] = val
