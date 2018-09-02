import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
for i in range(1000):
	env.render()
	observation, reward, done, info = env.step(env.action_space.sample())
	# observation, reward, done, info = env.step(i % 2)
	if done:
		pass
		# env.reset()

