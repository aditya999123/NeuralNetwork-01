from Data import Data
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
	data = Data()
	training_X, training_Y, test_X, test_Y = data.split(.2)

	# nn = NeuralNetwork()