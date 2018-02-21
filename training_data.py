import gym
import numpy as np

env = gym.make('CartPole-v0')

num_games = 10000
accepted_score = 50

accepted_runs = []
training_data = []
for _ in range(num_games):
	env.reset()
	score = 0

	prev_observation = []
	game_memory = []
	while True:
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)

		if len(prev_observation):
			game_memory.append([prev_observation, action])

		prev_observation = observation
		score += reward

		if done:
			break

	if score > accepted_score:
		for data in game_memory:
			if data[1] == 1:
				output = np.array([0, 1])
			elif data[1] == 0:
				output = np.array([1, 0])

			training_data.append([data[0], output])

training_data = np.array(training_data)
np.save('data.npy', training_data)

