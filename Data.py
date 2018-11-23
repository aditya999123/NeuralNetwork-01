import os, glob, pickle, tarfile
import numpy as np
import matplotlib.pylab as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

class Data:
	def __init__(self):
		self.data = {}
		self.training_X = None
		self.training_Y = np.zeros((1, 10))
		self.test_X = None
		self.test_Y = np.zeros((1, 10))

		self.read()

	def read(self):
		if os.path.isfile("./data.pkl"):
			print("loading data, can take few seconds...\n")
			with open('data.pkl', 'rb') as handle:
		   		self.data = pickle.load(handle)
			return

		# extracting from zip
		if not os.path.isdir("./trainingSet"):
			print("extracting from zip...\n")

			tf = tarfile.open("./trainingSet.tar.gz")
			tf.extractall()

		print("reading all files, can take few minutes...\n")

		for num in range(10):
			path = "%s/trainingSet/%d/*.jpg"%(dir_path, num)

			file_paths = glob.glob(path)
			print("digit: %d ... files: %d"%(num, len(file_paths)))

			x = np.zeros((len(file_paths), 784))
			# x = np.zeros((2, 784))

			for ind, file in enumerate(file_paths):
				image = plt.imread(file_paths[0])
				image = image.reshape(1, -1)

				x[ind:] = image

			self.data[num] = x

		with open('data.pkl', 'wb') as file:
		    pickle.dump(self.data, file, protocol=pickle.HIGHEST_PROTOCOL)

	def split(self, ratio = 0.1):
		training = {}
		test = {}

		for y, x in self.data.items():
			np.random.shuffle(x)
			num_files = x.shape[0]

			test_size = int(num_files * (1 - ratio))

			training[y] = x[:test_size]
			test[y] = x[test_size:]

		self.training_X = np.vstack((training[y] for y in training.keys()))
		self.test_X = np.vstack((test[y] for y in test.keys()))

		for y in training.keys():
			tmp = np.zeros((1, 10))
			tmp[0][y] = 1
			for i in range(training[y].shape[0]):
				self.training_Y = np.vstack((self.training_Y, tmp))

		for y in test.keys():
			tmp = np.zeros((1, 10))
			tmp[0][y] = 1
			for i in range(test[y].shape[0]):
				self.test_Y = np.vstack((self.test_Y, tmp))

		self.training_Y = self.training_Y[1:]
		self.test_Y = self.test_Y[1:]

		self.training_Y = self.training_Y.T
		self.test_Y = self.test_Y.T

		print("\ntraining_X: ", self.training_X.shape)
		print("training_Y: ", self.training_Y.shape)

		print("test_X: ", self.test_X.shape)
		print("test_Y: ", self.test_Y.shape)

		return self.training_X, self.training_Y, self.test_X, self.test_Y

data = Data()
data.split()