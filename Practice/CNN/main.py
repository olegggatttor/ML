import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(mnist_train_images, mnist_train_marks), (mnist_test_images, mnist_test_marks) = mnist.load_data()

mnist_train_images = mnist_train_images / 255.0
mnist_test_images = mnist_test_images / 255.0

mnist_train_images = mnist_train_images.reshape((mnist_train_images.shape[0], 28, 28, 1))
mnist_test_images = mnist_test_images.reshape((mnist_test_images.shape[0], 28, 28, 1))

fashion_mnist = keras.datasets.fashion_mnist

(fashion_mnist_train_images, fashion_mnist_train_marks), (
fashion_mnist_test_images, fashion_mnist_test_marks) = fashion_mnist.load_data()

fashion_mnist_train_images = fashion_mnist_train_images / 255.0
fashion_mnist_test_images = fashion_mnist_test_images / 255.0

fashion_mnist_train_images = fashion_mnist_train_images.reshape((fashion_mnist_train_images.shape[0], fashion_mnist_train_images.shape[1], fashion_mnist_train_images.shape[2], 1))
fashion_mnist_test_images = fashion_mnist_test_images.reshape((fashion_mnist_test_images.shape[0], fashion_mnist_test_images.shape[1], fashion_mnist_test_images.shape[2], 1))

models = [
    keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ]),
    keras.Sequential([
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5)),
        keras.layers.MaxPool2D(pool_size=(5, 5)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ]),
    keras.Sequential([
        keras.layers.Conv2D(filters=16, kernel_size=(2, 2)),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5)),
        keras.layers.MaxPool2D(pool_size=(5, 5)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ]),
    keras.Sequential([
        keras.layers.Conv2D(filters=8, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=16, kernel_size=(4, 4)),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ]),
    keras.Sequential([
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
]

best_acc = 0.0
best_loss = 100.0
best_model = models[0]
for model in models:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(mnist_train_images, mnist_train_marks, epochs=5)

    test_loss, test_acc = model.evaluate(mnist_test_images, mnist_test_marks)
    if best_acc < test_acc:
        best_acc = test_acc
        best_loss = test_loss
        best_model = model

print("Best accuracy: " + str(best_acc))
print("Best loss: " + str(best_loss))
best_model.summary()

best_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

best_model.fit(fashion_mnist_train_images, fashion_mnist_train_marks, epochs=5)
# test_loss, test_acc = best_model.evaluate(fashion_mnist_test_images, fashion_mnist_test_marks)

probs = best_model.predict(fashion_mnist_test_images)
classes = list(map(np.argmax, probs))
print(tf.math.confusion_matrix(fashion_mnist_test_marks, classes, num_classes=10))

res = list(zip(range(len(fashion_mnist_test_images)), fashion_mnist_test_marks, probs))

image_matrix = [[(0, 0)] * 10 for _ in range(10)]

for image_index, i, probabilities_for_image in res:
    for j in range(10):
        if image_matrix[i][j][0] < probabilities_for_image[j]:
            image_matrix[i][j] = (probabilities_for_image[j], image_index)

for row in image_matrix:
    print(" ".join(list(map(str, row))))


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure(figsize=(10,10))
for i in range(10):
    for j in range(10):
        plt.subplot(10, 10, i*10 + j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(fashion_mnist_test_images[image_matrix[i][j][1]], cmap=plt.cm.binary)
        if j == 0:
            plt.ylabel(class_names[i])
        if i == 9:
            plt.xlabel(class_names[j])
plt.show()
