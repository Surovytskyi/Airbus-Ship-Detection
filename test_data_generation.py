from tensorflow.python.keras.utils.data_utils import Sequence

from helper_functions import *

class TestDataGenerator(Sequence):

    def __init__(self, list_IDs, ship_seg_df, img_path_prefix, batch_size=32, dim=(32, 32), split_to_sub_img=True,
                 n_channels=3, shuffle_on_every_epoch=True, shuffle_on_init=True, forced_len=0,
                 drop_duplicates=False):
        # Initialization
        self.dim = dim  # dataset's dimension
        self.img_prefix = img_path_prefix  # location of the dataset
        self.batch_size = batch_size  # number of data/epoch
        self.ship_seg_df = ship_seg_df.copy()  # a dataframe storing the filenames and ship masks
        if drop_duplicates:
            self.list_IDs = np.unique(list_IDs)  # a list containing image names to be used by the generator
        else:
            self.list_IDs = list_IDs  # a list containing image names to be used by the generator
        self.n_channels = n_channels  # number of rgb chanels

        self.forced_len = forced_len  # Needed due to a bug in predict_generator

        # defines how many sub images should be generated. 50% overlap is used, so this should be an odd number
        # self.sub_img_ratio = split_img_ratio

        self.split_to_sub_img = split_to_sub_img
        #  Due to the 50% overlap of the number of sub images per whole image is:
        self.sub_img_count = int((768.0 / dim[0])) ** 2 + int(((768.0 / dim[0]) - 1)) ** 2
        self.sub_img_idx = 0
        self.sub_img_loc = [0, 0]

        # shuffle the data on after each epoch so data is split into different batches in every epoch
        self.shuffle_on_every_epoch = shuffle_on_every_epoch
        if shuffle_on_init:
            self.shuffle_data()
        else:
            self.indexes = np.arange(len(self.list_IDs))

        self.list_IDs_temp = []

    def on_epoch_end(self):
        if self.shuffle_on_every_epoch:
            self.shuffle_data()

        self.sub_img_idx = (self.sub_img_idx + 1) % self.sub_img_count

        self.sub_img_loc[0] = self.sub_img_loc[0] + self.dim[0]
        if self.sub_img_loc[0] + self.dim[0] > 768:
            # new row
            if self.sub_img_idx >= int((768.0 / self.dim[0])) ** 2 - 1:
                self.sub_img_loc[0] = int(self.dim[0] * 0.5)
            else:
                self.sub_img_loc[0] = 0

            self.sub_img_loc[1] = self.sub_img_loc[1] + self.dim[1]

        if self.sub_img_loc[1] + self.dim[1] > 768:
            if self.sub_img_idx >= int((768.0 / self.dim[0])) ** 2 - 1:
                self.sub_img_loc[0] = int(self.dim[0] * 0.5)
                self.sub_img_loc[1] = int(self.dim[1] * 0.5)
            else:
                # restart from the first corner
                self.sub_img_loc = [0, 0]

        # print(self.sub_img_loc, self.sub_img_idx)

    def shuffle_data(self):
        # Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batches per epoch'
        if self.forced_len == 0:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return self.forced_len

    def __getitem__(self, index):
        # Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.generate(self.list_IDs_temp)
        return X, Y

    def get_last_batch_ImageIDs(self):
        return self.list_IDs_temp

    def generate(self, tmp_list):
        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(tmp_list):
            # read image and mask referred by "ID"
            mask_list = self.ship_seg_df['EncodedPixels'][self.ship_seg_df['ImageId'] == ID].tolist()
            mask = np.zeros(self.dim)
            for mask_coded in mask_list:
                mask += rle_decode(mask_coded, shape=self.dim)

            img = read_transform_image(self.img_prefix + "/" + ID)

            X[i] = img
            Y[i] = np.atleast_3d(mask)

        return X, Y