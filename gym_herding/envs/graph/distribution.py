"""
Distribution Test Class

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import Union

import numpy as np

from gym_herding.envs.utils.position import to_matrix


class Distribution():
    """
    Distribution Class

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The number of agents to herd.
    weights : list[float, float]
        The weights applied to the population distribution within the state
        space.

    """
    def __init__(self, n_v: int, n_p: int, weights: List[float]) -> None:
        self._dist = {
            "initial": np.zeros((n_v, n_v)),
            "current": np.zeros((n_v, n_v)),
            "target": np.zeros((n_v, n_v)),
        }

        if isinstance(weights, (np.ndarray, list)) and len(weights) == 2:
            # Immutable parameter variables
            self._param = {
                "n_v": n_v,
                "n_p": n_p,
                "weights": weights,
                "count": np.array([0, 0]),
                "size_fraction": 1 / n_p,
            }

        else:
            raise ValueError("weights must be a 1x2 np.ndarray or list")

        self._status = {
            "initial": False,
            "target": False,
        }

    @property
    def initial(self) -> np.ndarray:
        """ Initial distribution getter property """
        return self._dist["initial"]

    @initial.setter
    def initial(self, matrix: np.ndarray) -> None:
        """ Initial distribution setter property """
        # if self._status["initial"]:
        #     raise ValueError("Initial Distribution already set.")

        # elif (isinstance(matrix, np.ndarray) and
        #       matrix.shape[0] == self._param["n_v"] and
        #       matrix.shape[1] == self._param["n_v"]):

        if (isinstance(matrix, np.ndarray) and
                matrix.shape[0] == self._param["n_v"] and
                matrix.shape[1] == self._param["n_v"]):

            if matrix.dtype != np.float32:
                raise TypeError("Initial must be a np.float32 np.ndarray.")

            if np.any(matrix):
                self._dist["initial"] = matrix
                self._dist["current"] = matrix
                self._status["initial"] = True

            else:
                raise ValueError("Must provide initial distribution " + \
                                 "of agents. A matrix with values between" + \
                                 " 0 and 1.")

        else:
            raise ValueError("Target distribution matrix should only " + \
                             "contain 1 or 0 values.")

    @property
    def current(self) -> np.ndarray:
        """ Current distribution getter property """
        return self._dist["current"]

    @current.setter
    def current(self, matrix: np.ndarray) -> None:
        """ Current distribution setter property """
        if matrix.dtype != np.float32:
            raise TypeError("Current must be a np.float32 np.ndarray.")

        self._dist["current"] = matrix

    @property
    def target(self) -> np.ndarray:
        """ Target distribution getter property """
        return self._dist["target"]

    @target.setter
    def target(self, matrix: np.ndarray) -> None:
        """ Target distribution setter property """
        if self._status["target"]:
            raise ValueError("Target Distribution already set.")

        elif (isinstance(matrix, np.ndarray) and
              matrix.shape[0] == self._param["n_v"] and
              matrix.shape[1] == self._param["n_v"]):

            if matrix.dtype != np.float32:
                raise TypeError("Target must be a np.float32 np.ndarray.")

            dist_info_0 = 0
            dist_info_1 = 0
            for i in range(0, matrix.shape[0]):
                for j in range(0, matrix.shape[1]):
                    if matrix[i, j] == 0:
                        dist_info_0 += 1

                    elif matrix[i, j] == 1:
                        dist_info_1 += 1

                    else:
                        raise ValueError("Target distribution matrix should" + \
                                         " only contain 1 or 0 values.")

            # Assign info
            self._dist["target"] = matrix
            self._param["count"][0] = dist_info_0
            self._param["count"][1] = dist_info_1
            self._status["target"] = True

        else:
            raise ValueError("Target distribution matrix should only " + \
                             "contain 1 or 0 values.")

    def get_node_value(self, i: int, j: int,
                       key: str = "target") -> Union[int, float]:
        """
        Get the distribution value for a specific node.

        The i,j coordinates must be in plotting form. The parameters will
        be transformed to matrix form for look-up in the distribuiton attribute.

        """
        if key not in self._dist:
            raise KeyError("key parameter must be a valid distribution: " +
                           "[\"initial\", \"current\", \"target\"]")

        elif not self._status["target"] or not self._status["initial"]:
            raise ValueError(
                "No initial or target distribution has been defined.")

        mat_i, mat_j = to_matrix(self._param["n_v"], np.array([i, j]))
        return self._dist[key][mat_i, mat_j]

    def set_node_value(self, val, i: int, j: int, key: str = "target") -> None:
        """
        Set the distribution value for a specific node.

        The i,j coordinates must be in plotting form. The parameters will
        be transformed to matrix form for look-up in the distribuiton attribute.

        """
        if key not in self._dist:
            raise KeyError("key parameter must be a valid distribution: " +
                           "[\"initial\", \"current\", \"target\"]")

        elif not self._status["target"] or not self._status["initial"]:
            raise ValueError(
                "No initial or target distribution has been defined.")

        mat_i, mat_j = to_matrix(self._param["n_v"], np.array([i, j]))
        self._dist[key][mat_i, mat_j] = val

    def increment_node_value(self, val: Union[int, float], i: int, j: int,
                             key: str = "target") -> None:
        """
        increment the distribution value for a specific node.

        The i,j coordinates must be in plotting form. The parameters will
        be transformed to matrix form for look-up in the distribuiton
        attribute.

        """
        if key not in self._dist:
            raise KeyError("key parameter must be a valid distribution: " +
                           "[\"initial\", \"current\", \"target\"]")

        elif not self._status["target"] or not self._status["initial"]:
            raise ValueError(
                "No initial or target distribution has been defined.")

        mat_i, mat_j = to_matrix(self._param["n_v"], np.array([i, j]))
        self._dist[key][mat_i, mat_j] += val * self._param["size_fraction"]

    def apply_population(self, key: str = "target") -> None:
        """ Apply population to distribution. """
        if key not in self._dist:
            raise KeyError("key parameter must be a valid distribution: " +
                           "[\"initial\", \"current\", \"target\"]")

        elif not self._status["target"] or not self._status["initial"]:
            raise ValueError(
                "No initial or target distribution has been defined.")

        if key == "target":
            temp_0 = self._param["weights"][0] * (1 / self._param["count"][0])
            temp_1 = self._param["weights"][1] * (1 / self._param["count"][1])

            for i in range(0, self._param["n_v"]):
                for j in range(0, self._param["n_v"]):
                    if self._dist[key][i, j] == 0:
                        self._dist[key][i, j] += temp_0
                    elif self._dist[key][i, j] == 1:
                        self._dist[key][i, j] *= temp_1

        elif key in ["initial", "current"]:
            temp = np.squeeze(np.where(self._dist[key] == 1))
            self._dist[key][temp] *= 1

    def get_agent_count(self, i: int, j: int, dist: str = "current") -> int:
        """ Get agent count for each state from distribution """
        return int(self._dist[dist][i, j] / self._param["size_fraction"])

    def reset(self) -> None:
        """ Reset certain mutable class properties """
        self._dist["current"] = np.copy(self._dist["initial"])
