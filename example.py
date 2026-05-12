import gymnasium as gym
from gym_torcs import TorcsEnv


# Launch TORCS automatically
env = TorcsEnv(throttle=True, max_episode_steps=10)


for ep in range(3):
    obs = env.reset(relaunch = True)
    done = False
    t = 0
    total_reward = 0
    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        print(t)
        t += 1 

    print(f"Episode {ep + 1} with reward: {reward}")

env.close()
