import gymnasium as gym
from gym_torcs import TorcsEnv


# Launch TORCS automatically
#env = TorcsEnv(render_mode="human", throttle=True, max_episode_steps=10000, port=3001)
env = gym.make(
    "TorcsSCR-v0",
    render_mode="human",        # or None
    executable="/usr/local/bin/torcs",
    port=3001,                  # 3001..3010
    track_name="street-1",
    track_category="road",
    laps=20,
    debug=True,
    gui_auto_start=True,
)

for ep in range(2):
    obs, info = env.reset()
    done = False
    t = 0
    total_reward = 0
    print(info) 
    while not done:
        action = env.action_space.sample()
        action[1] = abs(action[1])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {ep + 1} with reward: {reward}")

env.close()
