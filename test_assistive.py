import gym, src.envs

env = gym.make('FeedingPR2-v1')
env.render()
observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample() # Get a random action
    observation, reward, done, info = env.step(action)