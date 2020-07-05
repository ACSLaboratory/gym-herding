"""
2D Agent Rendering View for Herding Experiment.

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import TypeVar

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gym_herding.envs.graph.graph import NodeGraph
from gym_herding.envs.graph.leader import Leader


# Used for documentation purposes only
PlotType = TypeVar('PlotType')


class HerdingEnvPlotting():
    """
    Plotting methods and data for OpenAI Herding Env class

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The population of agents to herd.

    """

    def __init__(self, n_v: int, n_p: int) -> None:
        self.fig = None
        self.axis = None
        self.plots = None
        self._param = {
            "n_v": n_v,
            "n_p": n_p,
            "offset": 0.5,
        }
        self._agents_x: np.ndarray = np.empty(n_p)
        self._agents_y: np.ndarray = np.empty(n_p)

    def create_figure(self) -> None:
        """Create figure for rendering."""
        plt.ion()
        self.fig = plt.figure(1)
        self.axis = self.fig.add_subplot(
            111, xlim=(0, self._param["n_v"]), ylim=(0, self._param["n_v"]))
        self.axis.grid(True)
        plt.xticks(np.linspace(0, self._param["n_v"], self._param["n_v"] + 1))
        plt.yticks(np.linspace(0, self._param["n_v"], self._param["n_v"] + 1))
        a_plt, = self.axis.plot([], [], 'bx', markersize=5)
        l_plt, = self.axis.plot([], [], 'r.', markersize=15)
        self.plots = [a_plt, l_plt]

    def render(
            self,
            graph: NodeGraph,
            leader: Leader,
            is_initial: bool = False) -> None:
        """Render the motion of the leader and agents on the plot."""
        def animate(i):
            """Animate procedure for Fraction option."""
            idx = 0
            for node in graph:
                node_i, node_j = node.position
                agent_count = node.agent_count

                # Convert matrix coor (i, j) to cartesian coor (x, y)
                node_x = node_j + self._param['offset']
                node_y = (self._param['n_v'] - node_i) - self._param['offset']

                self._agents_x[idx:idx + agent_count] = self._get_visual_position(
                    node_x, agent_count)
                self._agents_y[idx:idx + agent_count] = self._get_visual_position(
                    node_y, agent_count)

                idx += agent_count

            self.plots[0].set_data(self._agents_x, self._agents_y)
            self.plots[1].set_data(leader.visual[0], leader.visual[1])
            return self.plots

        # Animate and draw new positions.
        if is_initial:
            anim = FuncAnimation(
                self.fig, animate, frames=1,
                init_func=self._init, repeat=False, blit=False)

        else:
            anim = FuncAnimation(
                self.fig, animate, frames=1, repeat=False,
                blit=False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self) -> None:
        """Reset certain mutable class properties."""
        raise NotImplementedError

    def save_render(self, file_name: str) -> None:
        """Save image of the render."""
        self.fig.savefig(file_name)

    def _init(self) -> List[PlotType]:
        """Initialize plotting of leader and agents."""
        self.plots[0].set_data([], [], 'bx', markersize=5)
        self.plots[1].set_data([], [], 'r.', markersize=15)
        return self.plots

    def _get_visual_position(self, point: int, pop_size: int) -> float:
        """Get x or y coordinate for visualization."""
        return point + np.random.uniform(
            -1 * self._param['offset'], self._param['offset'], size=pop_size)
