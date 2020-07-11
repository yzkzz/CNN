import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
classifier.add(Dense( units= 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense( units= 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 62.5)

res = classifier.evaluate(test_set,batch_size=25)  # 返回损失和精度

print(classifier.metrics_names);

print(res)

prob = classifier.predict(test_set)
print(np.argmax(prob))
