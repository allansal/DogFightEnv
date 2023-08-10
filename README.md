# DogFightEnv
A simple environment for obtaining explainable reinforcement learning (XRL) built with the Gymnasium library (https://gymnasium.farama.org/index.html). Read the included PDF file in the repository to get a more detailed explanation on the code contained in each file.

# Installing
## Dependencies
First, ensure you have the following packages installed:
1. **Gymnasium**: https://pypi.org/project/gymnasium/
2. **PyTorch**: https://pypi.org/project/torch/
3. **torchrl**: https://pypi.org/project/torchrl/
4. **PyGame**: https://pypi.org/project/pygame/

These can be installed using the Python pip package manger (https://pip.pypa.io/en/stable/) using the following command in your terminal:

`pip install gymnasium torch torchrl pygame`

## XRL DogFight Environment
Similar to other Gym / Gymnasium environments, this environment should be installed as a local pip package. To do this, you can run this command in your terminal:

`pip install -e ./gym-env`

# Running
Once you have installed all the dependencies and locally installed the Gymnasium environment, you can run this program using Python as follows:

`python DQN_DogFight.py`

By default, this will train an agent in a headless environment. You must modify the behavior of the code using command-line arguments. To display all the available parameters in the terminal, run:

`python DQN_DogFight.py --help`

As an example, say you want to load checkpoint 1000 in the ./checkpoints/test_run/ directory, then you would enter the command:

`python DQN_DogFight.py --checkpoint-dir test_run --load-checkpoint 1000`


A summary of all the parameters is provided in the table below.

| Parameter | Description |
| --- | --- |
| --help | Show the help message and exit |
| --render | Enable rendering of the environment to a PyGame window |
| --evaluate | Disable training and run the agent in evaluation mode |
| --checkpoint-dir | The subdirectory within the `checkpoints` directory to save or load checkpoints |
| --load-checkpoint | The checkpoint number to load |
| --checkpoint-interval | Episodic training interval for saving checkpoints |
| --exploration-episodes | Number of pure exploration episodes to perform (agent takes fully random actions) |
| --num-episodes | Number of episodes to train or evaluate for |
| --lr | The learning rate of the agent |
| --gamma | Discount factor for future rewards |
| --eps-max | Starting / maximum value for epsilon-greedy method |
| --eps-min | Final / minimum value for epsilon-greedy method |
| --memory-size | Size of the prioritized replay buffer |
| --batch-size | Replay buffer sample batch size |
| --per-alpha | Alpha value for prioritized experience replay |
| --per-beta | Beta value for prioritized experience replay |
| --per-eps | Epsilon value for prioritized experience replay |
| --tau | Soft-update factor between the policy and target networks |
| --seed | Random seed to use |
| --episode-time | Episode time limit (in seconds, at least based on human rendering mode) |
