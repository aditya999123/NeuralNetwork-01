from Data import Data
from NeuralNetwork import NeuralNetwork
import pickle

if __name__ == "__main__":
	data = Data()
	training_X, training_Y, test_X, test_Y = data.split(.1)

	nn = NeuralNetwork([784, 128, 128, 10])
	nn.train(training_X.T, training_Y, 1000, .01)

	with open('trained.pkl', 'wb') as file:
		pickle.dump(nn, file, protocol = pickle.HIGHEST_PROTOCOL)

	nn.predict(test_X.T, test_Y)
