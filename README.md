# gym-herding
Leader-agent herding OpenAI Gym Environment


## Setup
### Installation

To install this gym environment, please clone the repository to the directory you desire:

```bash
~$ git clone https://github.com/acslaboratory/gym-herding
```

Next install the environment using pip like this:

```bash
~$ python3 -m pip install -e gym-herding
```

### Uninstall

If you wish the uninstall the environment, you can simply do so using `pip`:

```bash
~$ python3 -m pip uninstall gym-herding
```

## Initialize the Environment

To initialize the default environment, use the following environment register:

```python
import gym

env = gym.make("gym-herding:Herding-v0")
```

Before using the environment, you must first initialize the environment using the `HerdingEnvParameter()` class as such:

```python
from gym_herding.env.parameters import HerdingEnvParameter

n_v = 2    # Square root of the graph size
n_p = 100  # Agent population size
weights = [8/10, 2/10]

hep = HerdingEnvParameter(n_v, n_p, weights)

env.initialize(hep)
```
