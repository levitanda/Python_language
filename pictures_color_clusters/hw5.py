from matplotlib import image
import numpy as np
from sklearn import cluster
from matplotlib import pyplot as plt


# A
def segement_image_with_kmeans(img, num_colors):
    x = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])
    km = cluster.KMeans(n_clusters=num_colors, random_state=42).fit(x)
    segm_image = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])
    labels = km.labels_
    centr = km.cluster_centers_
    for row in range(len(x)):
        label = labels[row]
        centroid = centr[label]
        segm_image[row] = centroid
    segm_image = segm_image.reshape(np.roll(img.shape, 0)).transpose(0, 1, 2)
    return segm_image


# B
def plot_segmented_images(original_img):
    plt.imshow(original_img)
    plt.axis('off')
    plt.title('original')
    plt.show()
    for k in [2, 4, 6, 8, 10]:
        image_ = np.copy(original_img)
        new_img = segement_image_with_kmeans(image_, k)
        plt.imshow(new_img)
        plt.axis('off')
        plt.title('There are ' + str(k) + ' colors')
        plt.show()


# C
def main_q_4():
    img1 = image.imread("beach.png")
    img2 = image.imread("ladybug.png")
    img3 = image.imread("road.png")
    plot_segmented_images(img1)
    plot_segmented_images(img2)
    plot_segmented_images(img3)


if __name__ == '__main__':
    main_q_4()
