import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

ship_dir = 'airbus-ship-detection-dataset'
train_image_dir = os.path.join(ship_dir, 'train_v2')
masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

#stratify by the number of boats appearing so for nice balances in each set

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# some files are too small/corrupt
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_image_dir,
                                                                                    c_img_id)).st_size/1024)
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50] # keep only 50kb files
masks.drop(['ships'], axis=1, inplace=True)

SAMPLES_PER_GROUP = 2000
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
#balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

# Split into train, validation and test groups
train_ids, valid_test_ids = train_test_split(balanced_train_df,
                 test_size = 0.3,
                 stratify = balanced_train_df['ships'])
test_ids, valid_ids = train_test_split(valid_test_ids,
                 test_size = 0.5)
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
test_df = pd.merge(masks, test_ids)

# Get dataset image names and labels
def get_ID_labels(df, labels=dict()):
    X_debug = df['ImageId'].values
    for i in range(X_debug.shape[0]):
        labels[X_debug[i]] = df.loc[df.ImageId == X_debug[i]].EncodedPixels.values
    return np.unique(X_debug), labels

train_ID, train_labels = get_ID_labels(train_df)
valid_ID, valid_labels = get_ID_labels(valid_df)
test_ID, test_labels = get_ID_labels(test_df)
