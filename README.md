# Gym-TORCS

## About this version

This implementation keeps the basic Gym-TORCS idea, but makes it a bit more useful for current RL experiments:

- updated to the new Gymnasium API
- track and road selection directly from the Python wrapper
- no manual TORCS config editing needed for normal experiments
- headless mode for faster training without the GUI
- optional GUI mode for debugging
- automatic GUI start when visual mode is used
- more stable reset handling by relaunching TORCS when needed
- support for multiple TORCS instances through different SCR ports
- configurable race type, track, laps, port, render mode and reset strategy
- simple continuous action space for steering and throttle/brake
- normalized observations
- optional observation and action noise
- useful info values such as speed, distance raced, off-track state and lap success

## References

Here are more TORCS references:

- [TORCS](https://torcs.sourceforge.net/)
- [Gym-TORCS (original)](https://github.com/ugo-nama-kun/gym_torcs)
- [vtorcs](https://github.com/giuse/vtorcs)
- [SnakeOil TORCS client](http://xed.ch/project/snakeoil/index.html)
- [TORCS for AI research paper](https://arxiv.org/pdf/1304.1672.pdf)

## Install

## Fedora Linux 44 Workstation

### Install TORCS 1.3.7 with the SCR server patch

1. Clone the TORCS repository:

```bash
git clone https://github.com/raphaelsenn/torcs-1.3.7
```

2. Enter the repository:

```bash
cd torcs-1.3.7
```

3. Install the required packages:

```bash
sudo dnf install \
  glib2-devel \
  mesa-libGL-devel \
  mesa-libGLU-devel \
  freeglut-devel \
  plib-devel \
  openal-soft-devel \
  freealut-devel \
  libXi-devel \
  libXmu-devel \
  libXrender-devel \
  libXrandr-devel \
  libpng-devel \
  libvorbis-devel \
  gcc \
  gcc-c++ \
  make \
  cmake \
  automake \
  autoconf \
  libtool \
  libXxf86vm-devel
```

4. Build and install TORCS:

```bash
make
sudo make install
sudo make datainstall
```

You should now be able to start TORCS with:

```bash
sudo torcs
```

### Install `gym_torcs`

1. Create and activate a conda environment:

```bash
conda create -n gym_torcs python=3.11 -y
conda activate gym_torcs
```

2. Clone the repository:

```bash
git clone https://github.com/raphaelsenn/gym_torcs
cd gym_torcs
```

3. Install the Python requirements:

```bash
pip install -r requirements.txt
```

4. Install `gym_torcs` in editable mode:

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

##  Citations

```bibtex
@misc{torcs,
  author       = {Espi{\'e}, Eric and Guionneau, Christophe and Wymann, Bernhard and others},
  title        = {{TORCS}: The Open Racing Car Simulator},
  year         = {2026},
  howpublished = {\url{https://sourceforge.net/projects/torcs/}},
  note         = {Accessed: 2026-05-15}
}

@misc{torcs137scr,
  author       = {Senn, Raphael},
  title        = {{TORCS} 1.3.7 with {SCR} Server Patch},
  year         = {2026},
  howpublished = {\url{https://github.com/raphaelsenn/torcs-1.3.7}},
  note         = {Patched version of TORCS 1.3.7 including the SCR server patch. Accessed: 2026-05-15}
}

@misc{gymtorcs,
  author       = {Senn, Raphael},
  title        = {{gym\_torcs}: A Gymnasium Interface for {TORCS}},
  year         = {2026},
  howpublished = {\url{https://github.com/raphaelsenn/gym_torcs}},
  note         = {Reinforcement learning environment for TORCS with a Gymnasium-like interface. Accessed: 2026-05-15}
}

@misc{loiacono2013scr,
  author       = {Loiacono, Daniele and Cardamone, Luigi and Lanzi, Pier Luca},
  title        = {Simulated Car Racing Championship: Competition Software Manual},
  year         = {2013},
  eprint       = {1304.1672},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/1304.1672}
}
```