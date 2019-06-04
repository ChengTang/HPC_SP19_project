import numpy as np
from mnist import MNIST
import os

mndata = MNIST(os.path.join(os.getcwd(), 'data/MNIST'))
training_data = mndata.load_training()

'''
# reshape images to 1d array
training_images = np.array(training_data[0]).reshape(-1,).astype(np.float32)

des = open("data/training_images.bin", "wb")
des.write(training_images)

training_labels = np.array(training_data[1]).reshape(-1,).astype(np.float32)

des = open("data/training_labels.bin","wb")
des.write(training_labels)
'''
des = open("data/training_images.bin", "wb")
training_images = (np.array(training_data[0]).reshape(-1,)).astype(np.float32).tofile(des,format = "%f")

des = open("data/training_labels.bin","wb")
training_labels = (np.array(training_data[1]).reshape(-1,)).astype(np.float32).tofile(des, format = "%f")


