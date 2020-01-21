"""
Agents Test Class

Written by: Zahi Kakish (zmk5)

"""
from typing import List

import numpy as np

from gym_herding.envs.distribution import Distribution


class Agents():
    """
    Agents Class

    Contains information on all the agents and not individual ones.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The number of agents to herd.
    weights : list[float, float]
        The weights applied to the population distribution within the state
        space.
    agent_type : {'individual', 'fraction'}, optional
        The distribution of agents can be rendered as individuals, or a
        fraction within a node/vertex.

    """
    def __init__(self, n_v: int, n_p: int, weights: List[float],
                 agent_type: str = "individual") -> None:
        # Immutable parameter variables
        self._param = {
            "n_v": n_v,
            "n_p": n_p,
            "weights": weights,
            "type": agent_type,
            "size_fraction": 1 / n_p,
        }

        if agent_type == "individual":
            self._real_positions = np.zeros((n_p, 2), dtype=np.int8)
            self._visual_positions = np.random.rand(n_p, 2) / n_v

        elif agent_type == "fraction":
            # For the fraction option, the _real_positions attribute only
            # stores the fraction of agents within node.
            self._real_positions = np.zeros(
                (np.power(n_v, 2), 2), dtype=np.int8)
            self._visual_positions = np.random.rand(n_p, 2) / n_v

        else:
            raise ValueError(
                "agent type must either be `individual` or `fraction`")

        self.distribution = Distribution(n_v, n_p, weights)

    @property
    def real(self) -> np.ndarray:
        """ Real Position getter property """
        return self._real_positions

    def set_real_pos(self, ind: int, pos: np.ndarray) -> None:
        """ Real Position setter for individual agent positions """
        if isinstance(pos, np.ndarray):
            self._real_positions[ind, :] = pos

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    @property
    def visual(self) -> np.ndarray:
        """ Visual Position Getter property """
        return self._visual_positions

    def set_visual_pos(self, ind: int, pos: np.ndarray) -> None:
        """ Visual Position setter for individual agent positions """
        if isinstance(pos, np.ndarray):
            self._visual_positions[ind, :] = pos

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    def reset(self) -> None:
        """ Reset certain mutable class properties """
        if self._param["type"] == "individual":
            self._real_positions = np.zeros(
                (self._param["n_p"], 2), dtype=np.int8)
            self._visual_positions = np.random.rand(
                self._param["n_p"], 2) / self._param["n_v"]

        else:
            self._real_positions = np.zeros(
                (np.power(self._param["n_v"], 2), 2), dtype=np.int8)
            self._visual_positions = np.zeros(
                (np.power(self._param["n_v"], 2), 2), dtype=np.int8)

        self.distribution.reset()
