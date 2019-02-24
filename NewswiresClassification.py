import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

def vectorize_sequences(sequences, dimensions=10000) :
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

print('____Preparing data____')

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print('____Building model____')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

print('____Training model____')

history = model.fit(x_train[1000:],
                    one_hot_train_labels[1000:],
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_train[:1000], one_hot_train_labels[:1000]))

print('____Evaluating model____')

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

predictions = model.predict(x_test)
print(predictions)