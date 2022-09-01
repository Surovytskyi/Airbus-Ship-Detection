from tensorflow.keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

from test_data_generation import *
from data_preparing import *
from options import test_options

test_opt = test_options()

# Load model
model = load_model('model.hdf5', compile=False)

# Define test data generator
test_generator = TestDataGenerator(
    test_ID, test_df,
    train_image_dir,
    batch_size=test_opt.batch_size,
    dim=test_opt.image_shape,
    shuffle_on_every_epoch=False,
    shuffle_on_init=False,
    split_to_sub_img=False,
    drop_duplicates=True
)
test_ratio = 0.01 #We choose the test partition too large, so we'll only use a smaller part of it
# Generate the first batch of images to initialize the variables
tmp_images, tmp_masks_true = test_generator.__getitem__(0)
tmp_ids = test_generator.get_last_batch_ImageIDs()
# Generate the predictions for the first batch
predictions_tmp = model.predict(tmp_images, verbose=test_opt.verbose)

# Initialize the variables storing the test data
test_images = tmp_images
test_ids = tmp_ids
test_mask_true = tmp_masks_true
predictions = predictions_tmp

for i in tqdm(range(1, int(test_ratio * len(test_ID) / test_opt.batch_size) + 1)):
    # Generate test images
    tmp_images, tmp_masks_true = test_generator.__getitem__(i)
    tmp_ids = test_generator.get_last_batch_ImageIDs()

    # Generate test predictions
    predictions_tmp = model.predict(tmp_images, verbose=test_opt.verbose)

    # Store everything
    test_images = np.concatenate((test_images, tmp_images), axis=0)
    test_ids = np.concatenate((test_ids, tmp_ids), axis=0)
    test_mask_true = np.concatenate((test_mask_true, tmp_masks_true), axis=0)
    predictions = np.concatenate((predictions, predictions_tmp), axis=0)

# Display test images and segmentation maps
for i in range(5):
    disp_image_with_map(test_images[i], test_mask_true[i], predictions[i])

# Confusion matrix
predictions = (predictions > test_opt.threshold).astype(int)
conf = confusion_matrix(test_mask_true.flatten(), predictions.flatten())
plot_confusion_matrix(conf, classes=["Non-ship", "Ship"], normalize=True)
plt.show()

# Test scores
balanced = balanced_accuracy_score(test_mask_true.flatten(), predictions.flatten())
f1 = f1_score(test_mask_true.flatten(), predictions.flatten())
dice = dice_coef_np(test_mask_true, predictions)
print("Balanced accuracy score:", balanced)
print("F1 similarity score: ", f1)
print("Dice coefficient", dice)