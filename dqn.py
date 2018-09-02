# 参考:https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import sys


class QFunction(chainer.Chain):
	def __init__(self, obs_size, n_actions, n_hidden_channels=50):
		super().__init__()
		with self.init_scope():
			self.l0 = L.Linear(obs_size, n_hidden_channels)
			self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
			self.l2 = L.Linear(n_hidden_channels, n_actions)
	
	def __call__(self, x, test=False):
		h = F.tanh(self.l0(x))
		h = F.tanh(self.l1(h))
		return chainerrl.action_value.DiscreteActionValue(self.l2(h))


def get_pure_agent():
	env = gym.make("CartPole-v0")
	print("Observation space:", env.observation_space)
	print("action space:", env.action_space)

	obs = env.reset()
	print("initial observation:", obs)

	action = env.action_space.sample()
	obs, r, done, info = env.step(action)
	print("next observation:", obs)
	print("reward:", r)
	print("done:", done)
	print("info:", info)

	obs_size = env.observation_space.shape[0]
	n_actions = env.action_space.n
	q_func = QFunction(obs_size, n_actions)

	# chainer.cuda.get_device(0).use()
	# q_func.to_gpu(0)

	_q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions, n_hidden_layers=2, n_hidden_channels=50)

	optimizer = chainer.optimizers.Adam(eps=1e-2)
	optimizer.setup(q_func)

	gamma = 0.95
	explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
	replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
	phi = lambda x: x.astype(np.float32, copy=False)
	agent = chainerrl.agents.DoubleDQN(
		q_func, optimizer, replay_buffer, gamma, explorer,
		replay_start_size=500, update_interval=1,
		target_update_interval=100, phi=phi
	)

	return agent


def train():
	env = gym.make("CartPole-v0")
	agent = get_pure_agent()

	print("\n[学習フェーズ]")

	n_episodes = 200
	max_episode_len = 200
	for i in range(1, n_episodes + 1):
		obs = env.reset()
		reward = 0
		done = False
		R = 0
		t = 0
		while not done and t < max_episode_len:
			# env.render()
			action = agent.act_and_train(obs, reward)
			obs, reward, done, info = env.step(action)
			R += reward
			t += 1
		if i % 10 == 0:
			print("episode:", i, "R:", R, "statistics:", agent.get_statistics())
		agent.stop_episode_and_train(obs, reward, done)
	print("Finished")
	agent.save("agent")
	return agent


def test():
	agent = get_pure_agent()
	agent.load("agent")
	print("\n[学習結果]")
	env = gym.make("CartPole-v0")
	for i in range(10):
		obs = env.reset()
		done = False
		R = 0
		t = 0
		while not done and t < 200:
			env.render()
			action = agent.act(obs)
			obs, r, done, info = env.step(action)
			R += r
			t += 1
		print("test episode:", i, "R:", R)
		agent.stop_episode()


def main():
	if len(sys.argv) < 2:
		mode = ""
	else:
		mode = sys.argv[1]
	if mode == "train":
		agent = train()
	elif mode == "test":
		test()
	else:
		train()
		test()


if __name__ == '__main__':
	main()