import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models, losses
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img
from matplotlib.pyplot import xticks, yticks


def plot_image_prediction(i, images, predictions, labels, class_names):
    plt.subplot(1, 2, 1)
    plt.imshow(images[i], cmap=plt.cm.binary)
    prediction = np.argmax(predictions[i])
    color = 'blue' if prediction == labels[i] else 'red'
    plt.title(f'{class_names[labels[i]]} (prognoza: {class_names[prediction]})', color=color)
    plt.subplot(1, 2, 2)
    plt.grid(False)
    plt.xticks(range(10))
    plot = plt.bar(range(10), predictions[i], color='#777777')
    plt.ylim([0, 1])
    plot[prediction].set_color('red')
    plot[labels[i]].set_color('blue')
    plt.show()

def generate_plot_pics(datagen, orginal_img, save_prefix):
    folder = 'aug_images'
    i = 0
    for batch in datagen.flow(orginal_img.reshape((1, 28, 28, 1)), batch_size=1, save_to_dir=folder, save_prefix=save_prefix, save_format='jpeg'):
        i += 1
        if i > 2:
            break
    plt.subplot(2, 2, 1, xticks=[], yticks=[])
    plt.imshow(orginal_img)
    plt.title("Orginał")
    i = 1
    for file in os.listdir(folder):
        if file.startswith(save_prefix):
            plt.subplot(2, 2, i + 1, xticks=[], yticks=[])
            aug_img = load_img(folder + '/' + file)
            plt.imshow(aug_img)
            plt.title(f'Uzupełniony {i}')
            i += 1
    plt.show()

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)
class_names = ['T-shirt/top', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz', 'Sandały', 'Koszula', 'Tenisówki', 'Torebka', 'Buty']
print(train_images.shape)
print(test_images.shape)
# plt.figure()
# plt.imshow(train_images[2])
# plt.colorbar()
# plt.grid(False)
# plt.title(class_names[train_labels[2]])
# plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.subplots_adjust(hspace=.3)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.title(class_names[train_labels[i]])
# plt.show()
X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))
print(X_train.shape)
tf.random.set_seed(42)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()
model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=10)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print('Dokładność dla zbioru testowego:', test_acc)
predictions = model.predict(X_test)
print(predictions[17])
print('Prognozowana etykieta próbki testowej: ', np.argmax(predictions[17]))
print('Rzeczywista etykieta próbki testowej: ', test_labels[17])
plot_image_prediction(17, test_images, predictions, test_labels, class_names)

# filters, _ = model.layers[2].get_weights()
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
# n_filters = 16
# for i in range(n_filters):
#     filter = filters[:, :, :, i]
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(filter[:, :, 0], cmap='gray')
# plt.show()
# datagen = ImageDataGenerator(horizontal_flip=True)
# generate_plot_pics(datagen, train_images[0], 'horizontal_flip')
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# generate_plot_pics(datagen, train_images[0], 'hv_flip')
# datagen = ImageDataGenerator(rotation_range=30)
# generate_plot_pics(datagen, train_images[0], 'rotation')
# datagen = ImageDataGenerator(width_shift_range=8)
# generate_plot_pics(datagen, train_images[0], 'width_shift')
# datagen = ImageDataGenerator(width_shift_range=8, height_shift_range=8)
# generate_plot_pics(datagen, train_images[0], 'width_height_shift')

# n_small = 500
# X_train = X_train[:n_small]
# train_labels = train_labels[:n_small]
# print(X_train.shape)
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=20, batch_size=40)
# test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
# print('Dokładność modelu dla zbioru testowego:', test_acc)
# datagen = ImageDataGenerator(height_shift_range=3, horizontal_flip=True)
# model_aug = tf.keras.models.clone_model(model)
# model_aug.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# train_generator = datagen.flow(X_train, train_labels, batch_size=40)
# model_aug.fit(train_generator, epochs=50, validation_data=(X_test, test_labels))
# test_loss, test_acc = model_aug.evaluate(X_test, test_labels, verbose=2)
# print('Dokładność modelu dla zbioru testowego:', test_acc)