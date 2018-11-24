import pickle
import numpy as np
import matplotlib.pylab as plt

with open('trained.pkl', 'rb') as file:
	nn = pickle.load(file)

fname = "test_images/" + "test4.jpg"
image = plt.imread(fname)
image = image.reshape(1, -1)
# image = image/255

co = np.zeros((1, 10))
co[0][8] = 1

my_predicted_image = nn.predict(image.T, co)
print("mpi", my_predicted_image)