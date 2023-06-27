import h5py
import numpy as np
from PIL import Image
import scipy
import logging
import seaborn as sns
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
# from vis.visualization import visualize_saliency
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import numpy as np
import os
import cv2 as cv

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense)
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K

original_classes = [
    [0, 23, 31],
    [1, 34, 40],  # !!!
    [2, 41, 47],
    [3, 48, 54],
    [4, 55, 61],
    [5, 62, 68],
    [6, 69, 75],
    [7, 76, 83],
    [8, 84, 90],
    [9, 91, 97],
    [10, 98, 105],
    [11, 106, 157],  # !!!
    [12, 175, 181],
    [13, 182, 188],
    [14, 189, 195],
    [15, 198, 204],
    [16, 205, 211],
    [17, 212, 218],
    [18, 230, 250],  # !!!
]

balanced_classes = [
    [0, 23, 37],
    [1, 38, 41],
    [2, 42, 45],
    [3, 46, 48],
    [4, 49, 52],
    [5, 53, 55],
    [6, 56, 63],
    [7, 64, 78],
    [8, 79, 93],
    [9, 94, 108],
    [10, 109, 157],  # !!!
    [11, 175, 181],
    [12, 182, 188],
    [13, 189, 204],
    [14, 205, 218],
    [15, 230, 250],  # !!!
]

large_classes = [
    [0, 23, 43],
    [1, 44, 50],
    [2, 51, 60],
    [3, 60, 70],
    [4, 71, 150],
    [5, 151, 200],
    [6, 201, 250]
]

predicted_classes = [
    [0, 23, 40],
    [1, 41, 42],
    [2, 43, 44],
    [3, 45, 47],
    [4, 48, 52],
    [5, 53, 59],
    [6, 60, 75],
    [7, 76, 101],
    [8, 102, 156],
    [9, 157, 188],
    [10, 189, 250]
]


def load_images_from_folder(folder):
    labels = []
    images = []
    days = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            aio_plant = filename.split("_")
            flowering_time = aio_plant[2].split(".")[0]
            for i in predicted_classes:
                if int(flowering_time) in range(i[1], i[2] + 1):
                    labels.append(np.uint8(i[0]))
                    images.append(np.asarray(img).astype(np.float32))
                    days.append(int(flowering_time))
    return np.asarray(images), np.asarray(days)


def build_model():
    classification_model = Sequential()
    classification_model.add(Conv2D(32, kernel_size=(5, 5), padding='same', strides=(1, 1), input_shape=(32, 32, 3),
                                    activation='relu'))

    classification_model.add(MaxPooling2D(pool_size=(2, 2)))

    classification_model.add(Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'))

    classification_model.add(MaxPooling2D(pool_size=(2, 2)))

    classification_model.add(Flatten(name='flatten'))

    classification_model.add(Dense(128, activation='relu'))
    classification_model.add(Dense(64, activation='relu'))

    classification_model.add(Dense(1, activation='linear'))

    classification_model.compile(optimizer='adam',
                                 loss='mean_squared_error',
                                 metrics=['mae'])
    classification_model.summary()

    return classification_model


'''
def plot_saliency(img_idx=None):
    img_idx = plot_features_map(img_idx)
    grads = visualize_saliency(cnn_saliency, -1, filter_indices=ytest[img_idx][0],
                               seed_input=x_test[img_idx], backprop_modifier=None,
                               grad_modifier="absolute")
    predicted_label = labels[np.argmax(cnn.predict(x_test[img_idx].reshape(1,32,32,3)),1)[0]]
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(x_test[img_idx])
    ax[0].set_title('original img id {} - {}'.format(img_idx, labels[ytest[img_idx][0]]))
    ax[1].imshow(grads, cmap='jet')
    ax[1].set_title('saliency - predicted {}'.format(predicted_label))
'''


def plot_confusion(confusion_mat):
    ax = sns.heatmap(confusion_mat, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels for all vigna\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    fig = ax.get_figure()
    fig.savefig('./confusion_matrix-all.png')
    plt.show()

    plt.matshow(confusion_mat)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_accuracy(history, fold):
    acc = history.history['mae']
    val_acc = history.history['val_mae']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(80)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Точность на обучении')
    plt.plot(epochs_range, val_acc, label='Точность на валидации')
    plt.legend(loc='lower right')
    plt.title('Точность')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Потери на обучении')
    plt.plot(epochs_range, val_loss, label='Потери на валидации')
    plt.legend(loc='upper right')
    plt.title('Потери')

    plt.savefig('./' + str(fold) + '_vigna_turned.png')
    plt.show()


def train():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    K.set_image_data_format('channels_last')

    # data_images, data_labels = load_images_from_folder('/Users/andrejbavykin/PycharmProjects/vigna_flowering/AIO_all')
    # train_images, train_labels = data_images[:1125], data_labels[:1125]
    # test_images, test_labels = data_images[1125:], data_labels[1125:]

    data_images, date_flowering_times = load_images_from_folder(
        'C:/Users/1/Desktop/AIO_vigna-master_updated/AIO_summer_square')
    train_images, train_flowering_times = data_images[:750], date_flowering_times[:750]
    test_images, test_flowering_times = data_images[750:], date_flowering_times[750:]

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_vectors = train_flowering_times.reshape(train_flowering_times.size, 1)
    test_vectors = test_flowering_times.reshape(test_flowering_times.size, 1)

    best_model = build_model()

    num_folds = 10
    batch_size = 64
    verbosity = 1
    acc_per_fold = []
    loss_per_fold = []

    data_images = data_images / 255.0
    data_vector = date_flowering_times.reshape(date_flowering_times.size, 1)
    print(data_vector)

    # Define the K-fold Cross Validator
    fold_no = 1
    kfold = KFold(n_splits=num_folds, shuffle=True)
    for tr, valid in kfold.split(train_images, train_flowering_times):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        history = best_model.fit(data_images[tr], data_vector[tr],
                                 batch_size=batch_size,
                                 epochs=80,
                                 verbose=verbosity,
                                 validation_split=0.2)
        plot_accuracy(history, fold_no)

        # Generate generalization metrics
        scores = best_model.evaluate(data_images[valid], data_vector[valid], verbose=0)
        print(
            f'Score for fold {fold_no}: {best_model.metrics_names[0]} of {scores[0]}; {best_model.metrics_names[1]} of {scores[1]}%')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1

        print(date_flowering_times[valid])
        print(np.argmax(best_model.predict(data_images[valid]), axis=-1))
        '''
        #cconfusion = confusion_matrix(data_labels[valid], np.argmax(best_model.predict(data_images[valid]), axis=-1))
        #print(cconfusion)
        #ax = sns.heatmap(cconfusion, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix with labels for all vigna\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        fig = ax.get_figure()
        fig.savefig('./confusion_matrix-val.png')
        plt.show()
        '''

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    '''
    confusion = confusion_matrix(test_labels, np.argmax(best_model.predict(test_images), axis=-1))
    print(confusion)

    test_acc = accuracy_score(test_labels, np.argmax(best_model.predict(test_images), axis=-1))
    print(test_acc)

    MCC = matthews_corrcoef(test_labels, np.argmax(best_model.predict(test_images), axis=-1))
    print(MCC)

    plot_confusion(confusion)
    '''
    # plot_saliency()



