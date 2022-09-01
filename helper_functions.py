import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import imageio
from tensorflow.keras.applications import imagenet_utils
import itertools
import tensorflow.keras.backend as K


cmap = pl.cm.viridis
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

def rle_decode(mask_rle, shape):

    # Decode mask to image

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    if mask_rle == mask_rle:  # if mask_rle is nan that this equality check returns false and the mask array remains 0
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1

    return mask.reshape(shape).T  # Needed to align to RLE direction

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply(y_true_f,y_pred_f))
    return (2.0 * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)

def preprocess_input(x):

    # Preprocesses a Numpy array encoding a batch of images.

    return imagenet_utils.preprocess_input(x, mode='tf')


def read_transform_image(img_file_path):

    # Image transformations
    img = imageio.imread(img_file_path)

    return preprocess_input(img)


def dice_coef(y_true, y_pred, smooth=1):

    # Dice score calculation
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(False)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def disp_image_with_map(img_matrix, mask_matrix_true, mask_matrix_pred, img_id=""):

    #Displays the image, the ground truth map and the predicted map

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_matrix * 0.5 + 0.5)
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("Image " + img_id)

    plt.subplot(1, 3, 2)
    plt.imshow(mask_matrix_true[:, :, 0], cmap='Greys')
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("Ground truth map")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_matrix_pred[:, :, 0], cmap='Greys')
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("Predicted map")

    plt.show()