# My first RL project
## View my report here: https://docs.google.com/document/d/1XbOEltMZHM-ia3lQGdXw1JRVAkRM4N84OaoYvlZgPMw/edit?usp=sharing

Made for an AI class assignment on Study.com.

This app trains an RL agent to reach target points by moving across a low-friction surface. It uses the [REINFORCE algorithm](https://en.wikipedia.org/wiki/Policy_gradient_method) with a baseline neural network and AdamW optimizers for agent training.

Uses `gymnasium` for environment, `torch` for neural net training, `pygame` for (admittedly minimal) environment rendering, and `matplotlib` for data viz.

Watch a trained agent in action [here](https://drive.google.com/file/d/1aH_fnqogJEjitqotSngLuN76zKu9J9C4/view?usp=sharing).

Example environment render:

<img width="483" height="512" alt="environment-window" src="https://github.com/user-attachments/assets/f1a8beb4-6780-4b5e-aadb-589f2fbd8c71" />

Example training graphs:

<img width="642" height="974" alt="Screenshot 2026-03-09 170445" src="https://github.com/user-attachments/assets/e5a4796a-9ad0-4c19-94e9-52cf06cc6175" />

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

