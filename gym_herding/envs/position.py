"""
Functions to manipulate position data.

Author: Zahi Kakish (zmk5)

"""
import numpy as np


def to_matrix(n_v: int, position: np.ndarray) -> np.ndarray:
    """
    Converts position data to matrix form.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    position : np.ndarray
        (x,y) position in plotting form. i.e. cartesian coordinates.

    Returns
    -------
    new_position : np.ndarray
        (i,j) position in matrix form. i.e. row-column matrix indexing.

    """
    if isinstance(position, np.ndarray):
        # Check if returning the Nx2 position array
        if position.shape[0] > 2 and position.shape[1] == 2:
            return np.array([(n_v - 1) - position[:, 1], position[:, 0]],
                            dtype=np.int8).T

    # else return 1x2 vector
    return np.array([(n_v - 1) - position[1], position[0]], dtype=np.int8)

def to_plot(n_v: int, position: np.ndarray) -> np.ndarray:
    """
    Converts position data to plotting form.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    position : np.ndarray
        (i,j) position in matrix form. i.e. row-column matrix indexing.

    Returns
    -------
    new_position : np.ndarray
        (x,y) position in plotting form. i.e. cartesian coordinates.

    """
    if isinstance(position, np.ndarray):
        # Check if returning the Nx2 position array
        if position.shape[0] > 2 and position.shape[1] == 2:
            return np.array([position[:, 1], (n_v - 1) - position[:, 0]],
                            dtype=np.int8).T

    # else return 1x2 vector
    return np.array([position[1], (n_v - 1) - position[0]],
                    dtype=np.int8)
