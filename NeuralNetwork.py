import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x)), x

def sigmoid_derivative(x):
	sigx, c = sigmoid(x)
	return sigx * (1 - sigx)

def relu(x):
	return x * (x > 0), x

def relu_derivative(x):
	if x==0:
		return 0

	return relu(x)/x

class NeuralNetwork:
	def __init__(self, layer_dimensions):
		self.params = {}

		for l in range(1, len(layer_dimensions)):
			self.params['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * .01
			self.params['b' + str(l)] = np.zeros((layer_dimensions[l], 1))

		self.L = len(self.params) // 2

	def linear_forward(self, A, W, b):
		Z = np.dot(W, A) + b

		return Z, (A, W, b)

	def linear_activation_forward(self, A_prev, W, b, activation):
		Z, linear_cache = self.linear_forward(A_prev, W, b)

		if activation == "sigmoid":
			A, activation_cache = sigmoid(Z)
		elif activation == "relu":
			A, activation_cache = relu(Z)

		cache = (linear_cache, activation_cache)

		return A, cache

	def forward(self, X):
		A = X
		caches = []
		L = self.L

		for l in range(1, L):
			A_prev = A
			A, cache = self.linear_activation_forward(A_prev, self.params["W" + str(l)], self.params["b" + str(l)], activation = "relu")
			caches.append(cache)

		AL, cache = self.linear_activation_forward(A, self.params["W" + str(L)], self.params["b" + str(L)], activation = "sigmoid")
		caches.append(cache)

		return AL, caches

	def cost(self, AL, Y):
		# print (AL)
		# print (Y)


		m = Y.shape[1]
		# print("m=", m)
		aa = np.multiply(np.log(AL), Y)
		# print("aa=", aa.shape)

		bb = np.multiply(np.log(1 - AL), (1 - Y))
		# print("bb=", bb.shape)
		cost = (-1/m)*(np.sum(aa + bb, axis = 0, keepdims = True))
		# print("cost=", cost.shape)
		# print("cost=", cost)

		# print("cost=", cost.shape)
		# print("cost=", cost)

		# exit(1)

		return cost

	def linear_backward(self, dZ, cache):
		A_prev, W, b = cache
		m = A_prev.shape[1]

		dW = (1/m) * np.dot(dZ, A_prev.T)
		db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
		dA_prev = np.dot(W.T, dZ)

		return dA_prev, dW, db

	def linear_activation_backward(self, dA, cache, activation):
		linear_cache, activation_cache = cache
		A, W, b = linear_cache

		if activation == "relu":
			dZ = np.multiply(dA, relu_derivative(activation_cache))
		elif activation == "sigmoid":
			dZ = np.multiply(dA, sigmoid_derivative(activation_cache))

		dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		return dA_prev, dW, db

	def backward(self, AL, Y, caches):
		m = AL.shape[1]
		dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

		L = self.L

		grads = {}
		Y = Y.reshape(AL.shape)

		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")

		for l in reversed(range(L - 1)):
			current_cache = caches[l]
			grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "sigmoid")

		return grads

	def update(self, grads, learning_rate):
		for l in range(self.L):
			self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
			self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	def train(self, training_X, training_Y, iterations, learning_rate):
		for i in range(iterations):
			AL, caches = self.forward(training_X)

			cost = self.cost(AL, training_Y)

			grads = self.backward(AL, training_Y, caches)

			self.update(grads, learning_rate)

			if i % 100 == 0:
				print("i = ", i, " cost= ", np.squeeze(np.sum(cost, axis=0))) 

	def predict(self, test_X, test_Y):
		AL, caches = self.forward(test_X)

		print("AL:", AL)
		output = (AL > 0.1).astype(int)
		m = output.shape[1]

		print(output)
		accuracy = np.sum(output == test_Y, axis = 0, keepdims = True)/output.shape[0]

		print(accuracy)
		acc = np.sum(accuracy, axis = 1)/output.shape[1]
		np.squeeze(acc)
		
		print(acc)

		return output