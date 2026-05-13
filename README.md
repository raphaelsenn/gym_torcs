# Gym-TORCS

Gym-TORCS is the reinforcement learning (RL) environment in TORCS domain with OpenAI-gym-like interface.
TORCS is the open-rource realistic car racing simulator recently used as RL benchmark task in several AI studies.

Gym-TORCS is the python wrapper of TORCS for RL experiment with the simple interface (similar, but not fully) compatible with OpenAI-gym environments. The current implementaion is for only the single-track race in practie mode. If you want to use multiple tracks or other racing mode (quick race etc.), you may need to modify the environment, "autostart.sh" or the race configuration file using GUI of TORCS.

This code is developed based on vtorcs (https://github.com/giuse/vtorcs)
and python-client for torcs (http://xed.ch/project/snakeoil/index.html).

The detailed explanation of original TORCS for AI research is given by Daniele Loiacono et al. (https://arxiv.org/pdf/1304.1672.pdf)

Because torcs has memory leak bug at race reset.
As an ad-hoc solution, we relaunch and automate the gui setting in torcs.
Any better solution is welcome!

## Requirements
We are assuming you are using Ubuntu 26.04 LTS or Fedora Workstation 44. The software below should be installed:
* Python 3.11
* xautomation (http://linux.die.net/man/7/xautomation)
* vtorcs-RL-color (installation of vtorcs-RL-color is explained in vtorcs-RL-color directory)

## Initialization of the Race
After the insallation of vtorcs-RL-color, you need to initialize the race setting. You can find the detailed explanation in a document (https://arxiv.org/pdf/1304.1672.pdf), but here I show the simple gui-based setting.

So first you need to run
```
sudo torcs
```
in the terminal, the GUI of TORCS should be launched.
Then, you need to choose the race track by following the GUI (Race --> Practice --> Configure Race) and open TORCS server by selecting Race --> Practice --> New Race. This should result that TORCS keeps a blue screen with several text information.

If you need to treat the vision input in your AI agent, you have to set the small image size in TORCS. To do so, you have to run
```
python snakeoil3_gym.py
```
in the second terminal window after you open the TORCS server (just as written above). Then the race starts, and you can select the driving-window mode by F2 key during the race.

After the selection of the driving-window mode, you need to set the appropriate gui size. This is done by using the display option mode in Options --> Display. You can select the Screen Resolution, and you need to select 64x64 for visual input (our immplementation only support this screen size, other screen size results the unreasonable visual information). Then, you need to shut down TORCS to complete the configuration for the vision treatment.

## Install

1. Create a virtual environment using conda and activate it:

```bash
conda create -n gym_torcs python=3.11 -y
conda activate gym_torcs
```

2. Clone the repository and move into the repository:

```bash
git clone https://github.com/raphaelsenn/gym_torcs
cd gym_torcs
```


3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Install gym_torcs

```bash
pip install -e .
```
## Usage

### Basic headless training example

The recommended mode for reinforcement learning is `render_mode=None`. In this mode, the environment starts TORCS in text/headless mode and connects to the SCR server automatically.

```python
import gymnasium as gym
import gym_torcs


env = gym.make(
    "TorcsSCR-v0",
    render_mode=None,
    port=3001,
    track_name="forza",
    track_category="road",
    throttle=True,
    reset_strategy="relaunch",
    debug=False,
)

obs, info = env.reset()

done = False
episode_reward = 0.0

while not done:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    episode_reward += reward

print(f"Episode reward: {episode_reward:.2f}")

env.close()
```

##  Acknowledgement
gym_torcs was developed during the spring internship 2016 at Preferred Networks.
