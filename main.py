import numpy as np
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
	print ("NeuralNetwork-101 started...")

	X = np.array([[0,0,1],
				  [0,1,1],
				  [1,0,1],
				  [1,1,1]])

	print X.shape[1]
	print np.random.rand(3,4) 
	y = np.array([[0],[1],[1],[0]])
	nn = NeuralNetwork(X,y)

	for i in range(1500):
		nn.feedforward()
		nn.backprop()

	print(nn.output)
