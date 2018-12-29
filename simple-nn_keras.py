"""
Simple NN Implementation in Keras.

Following tutorials from Deep Learning with Python (Francois Chollet)
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb, reuters
from keras import models, layers


num_words = 10000
def imdb_review_sentiment(num_words):
    """
    Classify IMDB film reviews as positive/negative.

    Uses two hidden layers, both with relu activations.
    Other params:

    Args:
        num_words tells how many unique words to load (gets rid of rare ones)

    The data:
        The data themselves are indices of words for a dictionary (word book) list
        Labels are 1=positive review, 0=negative review.

    More on p. 68 of the book.
    """
    # Load the data.
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    train_data[0]


    # Want to prepare data to be constant-shape tensors of shape (samples, word_indices)
    def vectorize_sequences(sequences, dimension=num_words):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results


    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Split out some training/validation data
    n_trainers = int(0.4 * len(train_data))

    x_val = x_train[: n_trainers]
    partial_x_train = x_train[n_trainers:]

    y_val = y_train[: n_trainers]
    partial_y_train = y_train[n_trainers :]


    # Cool, we've got the data formatted right, now set up the model.
    # The number (16, 16, and 1) is how many elements long the output is? p.70
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Added the layers, now compile the model.
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # The results vector:
    results = model.evaluate(x_test, y_test)

    # We can also do some predictions:
    model.predict(x_test)

    # Train the model. This is the one that takes time.
    history = model.fit(partial_x_train, partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val)
                        )

    # Check out the results
    history_dict = history.history
    acc_values = history_dict['acc']
    loss_values = history_dict['loss']
    val_acc_values = history_dict['val_acc']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, loss_values, 'ob', label='Training Loss')
    plt.plot(epochs, val_loss_values, '-r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc_values, 'ob', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, '-r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return results

def reuters_topic_classification(num_words):
    """
    Classify Reuters newswires into one of 46 topics.

    Section 3.5, p. 79
    """
    # Load the data.
    (train_data, train_labels), (test_data, test_labels) = reutere.load_data(num_words=num_words)

    # Want to prepare data to be constant-shape tensors of shape (samples, word_indices)
    def vectorize_sequences(sequences, dimension=num_words):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    def to_categorical(labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label = in enumerate(labels):
            results[i, label] = 1
        return results

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]


    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(num_words,)))
    model.add(layers.Dense(64, activation='relu'))
    # Softmax outs a 46-dimensional probably vector whose elements sum to 1.
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    results = model.evaluate(x_test, one_hot_test_labels)

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss_values, 'ob', label='Training Loss')
    plt.plot(epochs, val_loss_values, '-r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc_values, 'ob', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, '-r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return results






# The End
