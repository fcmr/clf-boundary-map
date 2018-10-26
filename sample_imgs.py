import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)

X_full = np.load('data/fashionmnist/X_full.npy')
y = np.load('data/fashionmnist/y_full.npy')

idx_all = np.arange(len(y))

selected = []
for i in range(10):
    idx_i = idx_all[y == i]
    selected.append(idx_i[random.randint(0, len(idx_i))])

print(selected)
selected = np.array(selected).reshape((5,2))

new_img = np.zeros((5*28, 2*28))
for i in range(5):
    for j in range(2):
        print(i, j)
        print(i*28, (i+1)*28, j*28,(j+1)*28)
        img = X_full[selected[i, j]].reshape((28,28))
        new_img[i*28:(i+1)*28, j*28:(j+1)*28] = img


plt.xticks([])
plt.yticks([])
plt.imshow(new_img, cmap='gray')
plt.imsave('dataset_samples.png', new_img, cmap='gray', format='png')
plt.show()


