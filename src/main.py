import config
import train

import os
from glob import glob

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

# Check If GPU is available
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# Count files in directories and subdirectories
def count(directory, counter=0):
    for pack in os.walk(directory):
        for _ in pack[2]:
            counter += 1
    return directory + " : " + str(counter) + " files"


# Calculate total number of images for training and testing
print('total images for training :', count(config.TRAIN_PATH))
print('total images for validation :', count(config.TEST_PATH))

# Calculates total number of classes
folders = glob('data/grayscale/train/*')
print(len(folders))

# Initialize train_generator and test_generator
train_generator = ImageDataGenerator(validation_split=0.1)
test_generator = ImageDataGenerator()

# Prepares Training and Testing Data
class_subset = sorted(os.listdir('../data/grayscale/train'))
print(class_subset)

traingen = train_generator.flow_from_directory(config.TRAIN_PATH,
                                               target_size=(256, 256),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='training',
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True,
                                               seed=5)

validgen = train_generator.flow_from_directory(config.TRAIN_PATH,
                                               target_size=(256, 256),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='validation',
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True,
                                               seed=4)

testgen = test_generator.flow_from_directory(config.TEST_PATH,
                                             target_size=(256, 256),
                                             class_mode=None,
                                             classes=class_subset,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=4)


train.train(traingen, validgen, testgen)
