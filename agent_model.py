import gym
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
import numpy as np 
import sys

load_model = True
if len(sys.argv) > 1:
	load_model = False

#Data and hyperparamters
training_data = np.load('data.npy')
input_size = training_data[0][0].shape[0]
output_size = training_data[0][1].shape[0]
keep_prob = 0.7
lr = 1e-3
num_epochs = 3

#Model
network = input_data(shape=[None, input_size, 1], name='x')

network = fully_connected(network, n_units=128, activation='relu')
network = dropout(network, keep_prob)

network = fully_connected(network, n_units=256, activation='relu')
network = dropout(network, keep_prob)

network = fully_connected(network, n_units=512, activation='relu')
network = dropout(network, keep_prob)

network = fully_connected(network, n_units=256, activation='relu')
network = dropout(network, keep_prob)

network = fully_connected(network, n_units=128, activation='relu')
network = dropout(network, keep_prob)

network = fully_connected(network, n_units=output_size, activation='softmax')
network = tflearn.layers.estimator.regression(network, optimizer='adam', 
					loss='categorical_crossentropy', learning_rate=lr, name='y_')

model = tflearn.DNN(network, tensorboard_dir='log')

#Train
if not load_model:
	x = np.array([i[0] for i in training_data]).reshape(-1, input_size, 1)
	y = np.array([i[1] for i in training_data])

	model.fit({'x': x}, {'y_': y}, n_epoch=num_epochs, snapshot_step=500, 
					show_metric=True, shuffle=True)
	model.save('model/cartpole_agent.model')


model.load('model/cartpole_agent.model')

#Play
env = gym.make('CartPole-v0')

for g in range(10):
	print "Game {}:".format(g)

	env.reset()
	score = 0
	prev_observation = []

	while True:
		env.render()

		if not len(prev_observation):
			action = env.action_space.sample()

		else:
			action = np.argmax(model.predict(prev_observation.reshape(-1, input_size, 1)))

		observation, reward, done, info = env.step(action)
		prev_observation = observation

		score += reward
		if done:
			print score
			break

