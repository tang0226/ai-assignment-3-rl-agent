# My first RL project
## View my report here: https://docs.google.com/document/d/1XbOEltMZHM-ia3lQGdXw1JRVAkRM4N84OaoYvlZgPMw/edit?usp=sharing

Made for an AI class assignment.

This app trains an RL agent to reach target points by moving across a low-friction surface. It uses the REINFORCE algorithm with a baseline function and AdamW optimizers for agent training.

Uses `gymnasium` for environment, `torch` for neural net training, `pygame` for environment rendering, and `matplotlib` for data viz.

Watch a trained agent in action [here](https://drive.google.com/file/d/1aH_fnqogJEjitqotSngLuN76zKu9J9C4/view?usp=sharing).


## Running the app
Clone this repo.

Ensure Python 3.13 is installed (https://www.python.org/downloads/). Pygame has not yet published a `pip` wheel for Python 3.14, so an earlier version is needed.

### Create a Python 3.13 virtual environment:
```bash
python3.13 -m venv .venv
```

### Activate the virtual environment:
MacOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

### Install required modules using `pip`:
```bash
pip install torch torchvision gymnasium pygame numpy matplotlib
```

### Run the app:
```bash
python main.py
```
