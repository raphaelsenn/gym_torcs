import gym
from gym_torcs import TorcsEnv


# Launch TORCS automatically
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
obs = env.reset(relaunch=True)
done = False

total_reward = 0
while not done:
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)

    total_reward += reward
    print(f'Reward: {reward}')

print("episode reward:", total_reward)

env.end()
