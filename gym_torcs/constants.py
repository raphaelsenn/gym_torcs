PORT = 3001
TERMINAL_JUDGE_START = 500      # Speed limit is applied after this step
TERMINATION_LIMIT_PROGRESS = 5  # In [km/h] (episode terminates if car is running slower than this limit)

DEFAULT_SPEED = 50              # In [km/h]
MAX_FOCUS = 200.0               # In [m]
MAX_OPPONENTS = 200.0           # In [m]
MAX_TRACK = 200.0               # In [m]

OBS_NOISE_STD = 0.05
ACTION_NOISE_STD = 0.05 