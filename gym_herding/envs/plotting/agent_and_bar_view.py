"""
2D Bar Graph Rendering View for Herding Experiment.

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


class PlotAgentAndBarView():
    """Agent and Bar plot for OpenAI Herding Env class.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The population of agents to herd.

    """

    def __init__(self, n_v: int, n_p: int) -> None:
        """Initialize the PlotAgentAndBarView class."""
        self.fig = None
        self.axis = {}
        self.plots = None
        self._param = {
            'n_v': n_v,
            'n_p': n_p,
            'target': np.zeros(n_v * n_v),
            'vertex_x_bar': list(range(1, n_v * n_v + 1)),
            'vertex_x_bar_labels': [str(i) for i in range(1, n_v * n_v + 1)],
            'offset': 0.5,
        }
        self._agents_x: np.ndarray = np.empty(n_p)
        self._agents_y: np.ndarray = np.empty(n_p)

    def create_figure(self) -> None:
        """Create figure for rendering."""
        plt.ion()
        self.fig = plt.figure(1)

        # Create subplot showing leader motion
        self.axis[0] = self.fig.add_subplot(
            121, xlim=(0, self._param["n_v"]), ylim=(0, self._param["n_v"]))
        self.axis[0].grid(True)
        self.axis[0].set_xticks(
            np.linspace(0, self._param["n_v"], self._param['n_v'] + 1))
        self.axis[0].set_yticks(
            np.linspace(0, self._param["n_v"], self._param['n_v'] + 1))
        a_plt, = self.axis[0].plot([], [], 'bx', markersize=5)
        l_plt, = self.axis[0].plot([], [], 'r.', markersize=15)

        # Create subplot for showing agent density
        self.axis[1] = self.fig.add_subplot(122)
        self.axis[1].set_xlim(0, self._param['n_v'] * self._param['n_v'] + 1)
        self.axis[1].set_ylim(0, 1)
        self.axis[1].set_xticks(
            self._param['vertex_x_bar'], self._param['vertex_x_bar_labels'])
        self.axis[1].set_yticks(np.linspace(0, 1, 6))
        b_target_plt = self.axis[1].bar(
            self._param['vertex_x_bar'],
            self._param['target'], color='red')
        b_current_plt = self.axis[1].bar(
            self._param['vertex_x_bar'],
            np.zeros(self._param['n_v'] * self._param['n_v']))

        # Put all animated plot features within a single list
        self.plots = [a_plt, l_plt, b_target_plt, b_current_plt]

    def render(self, graph: NodeGraph, leader: Leader,
               is_initial: bool = False) -> None:
        """Render the motion of the leader and agents on the plot."""
        def animate(i):
            """ Animation procedure for Fraction option """
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

            # Set leader and Agent motion on subplot data
            self.plots[0].set_data(self._agents_x, self._agents_y)
            self.plots[1].set_data(leader.visual[0], leader.visual[1])

            # Get target and current agent density bar graph data
            target_plot_data = []
            for j in range(0, self._param['n_v']):
                for k in range(0, self._param['n_v']):
                    target_plot_data.append(graph.distribution.target[j, k])

            bar_plot_data = []
            for node in graph:
                agent_density = node.agent_count / self._param['n_p']
                bar_plot_data.append(agent_density)

            # Set target and current agent density bar graph data
            self.axis[1].clear()
            self.axis[1].set_xlim(0, self._param['n_v'] * self._param['n_v'] + 1)
            self.axis[1].set_ylim(0, 1)
            self.axis[1].set_xticks(
                self._param['vertex_x_bar'], self._param['vertex_x_bar_labels'])
            self.axis[1].set_yticks(np.linspace(0, 1, 6))
            self.plots[2] = self.axis[1].bar(
                np.array(self._param['vertex_x_bar']) - 0.35/2,
                target_plot_data, 0.35, color='red')
            self.plots[3] = self.axis[1].bar(
                np.array(self._param['vertex_x_bar']) + 0.35/2,
                bar_plot_data, 0.35)
            self.axis[1].legend(['Target', 'Current'])

            # return new data to FuncAnimation
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
