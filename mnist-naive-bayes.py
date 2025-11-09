from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mnist = fetch_openml('mnist_784')

X = mnist.data.values
Y = mnist.target.values.astype(np.int16)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=1/7, random_state=42, shuffle=True)


n_class = len(mnist.target.unique())
n_pixels = x_train.shape[-1]
matrix_pixel_class_mean_var = np.zeros((n_pixels, n_class, 2))
p_class = np.ones(n_class) * (1/n_class)

# learning the distribution from training data
for class_idx in range(n_class):
    for pixel_idx in range(n_pixels):
        pixel_class_vec = x_train[y_train == class_idx , pixel_idx]
        matrix_pixel_class_mean_var[pixel_idx, class_idx] = [pixel_class_vec.mean(), pixel_class_vec.var()]


# ploting learned distributions for each class
for i in range(n_class):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Class = {i}')
    pixel_vector = matrix_pixel_class_mean_var[:, i, 0]
    plt.imshow(pixel_vector.reshape(28,28))
plt.show()


def guassian_function(x, mean, var, smoothing):
    var = var + smoothing
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(x - mean)/(2 * var))


i = 0
predictions = np.zeros(x_test.shape[0])
for x in tqdm(x_test):
    plt.imshow(x.reshape(28,28))
    plt.show()
    result = np.zeros(n_class)
    for class_idx in range(n_class):
        p = np.log(p_class[class_idx])
        for pixel_idx in range(n_pixels):
            mean, var = matrix_pixel_class_mean_var[pixel_idx, class_idx]
            p+= np.log(guassian_function(x[pixel_idx], mean, var, 1000))
        result[class_idx] = p
    pred = np.argmax(result)
    predictions[i] = pred 
    print(pred)
    i +=1


print(classification_report(y_test, predictions))

