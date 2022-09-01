from tensorflow.python.keras.utils.data_utils import Sequence
import cv2
import os

from helper_functions import *

class TrainDataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, img_path, batch_size, dim, shuffle=True,
                 augmentation=None):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.img_path = img_path
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()
        self.list_IDs_temp = []

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Initialization
        batch_x = np.zeros((self.batch_size,) + (*self.dim, 3))
        batch_y = np.zeros((self.batch_size,) + (*self.dim, 1))
        for i, img_id in enumerate(self.list_IDs_temp):
            # Generate data
            X, y = self.generation(img_id)
            batch_x[i] += X
            batch_y[i] += y

        return batch_x, batch_y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def generation(self, img_id):
        # Generates data containing batch_size samples  X : (n_samples, *dim, n_channels)
        # Initialization

        # Generate data
        image = cv2.cvtColor(cv2.imread(os.path.join(self.img_path, img_id)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.dim)


        mask = np.zeros((*self.dim, 1))
        for mask_rle in self.labels[img_id]:
            if mask_rle is np.nan:
                continue
            # Store masks
            mask[:, :, 0] += rle_decode(mask_rle, shape=self.dim)

        # Image and mask transformation
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask