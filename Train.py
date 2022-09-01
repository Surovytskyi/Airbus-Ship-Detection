from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_preparing import *
from train_data_generation import *
from options import train_options

train_opt = train_options()

# Import augmentation library for train data extension
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Transpose,
    OneOf, ToFloat, RandomGamma, CLAHE, ElasticTransform, GaussNoise, HueSaturationValue,
    Blur, MotionBlur, MedianBlur, RandomBrightnessContrast, GridDistortion, OpticalDistortion, RandomSizedCrop)

# Define augmentation tool
augmentor = Compose([
    OneOf([HorizontalFlip(), VerticalFlip(), RandomRotate90(), Transpose()], p=0.8),
    #ShiftScaleRotate(rotate_limit=20),
    OneOf([MotionBlur(), MedianBlur(), Blur()], p=0.3),
    OneOf([RandomGamma(), RandomBrightnessContrast(), CLAHE()], p=0.3),
    OneOf([HueSaturationValue(), GaussNoise()], p=0.2),
    OneOf([ElasticTransform(), OpticalDistortion(), GridDistortion()], p=0.3),
    RandomSizedCrop(min_max_height=(train_opt.image_shape[0]/2, train_opt.image_shape[0]),
                    height=train_opt.image_shape[0], width=train_opt.image_shape[1], p=0.3),
    ToFloat(max_value=1), ], p=1)


# Define model
def Unet_encoder_layer(input_layer, kernel, filter_size, pool_size):
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    residual_connection = x
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    return x, residual_connection


def Unet_decoder_layer(input_layer, kernel, filter_size, pool_size, residual_connection):
    filter_size = int(filter_size)
    x = UpSampling2D(size=(pool_size, pool_size))(input_layer)
    x = Concatenate()([residual_connection, x])
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def Unet(input_layer):

    residual_connections = []

    x, residual_connection = Unet_encoder_layer(input_layer, train_opt.kernel, train_opt.filter_size, train_opt.pool_size)
    residual_connections.append(residual_connection)

    train_opt.filter_size *= 2
    x, residual_connection = Unet_encoder_layer(x, train_opt.kernel, train_opt.filter_size, train_opt.pool_size)
    residual_connections.append(residual_connection)

    train_opt.filter_size *= 2
    x, residual_connection = Unet_encoder_layer(x, train_opt.kernel, train_opt.filter_size, train_opt.pool_size)
    residual_connections.append(residual_connection)


    train_opt.filter_size *= 2
    x = Conv2D(train_opt.filter_size, train_opt.kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(train_opt.filter_size, train_opt.kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)


    train_opt.filter_size /= 2
    x = Unet_decoder_layer(x, train_opt.kernel, train_opt.filter_size, train_opt.pool_size, residual_connections[-1])
    residual_connections = residual_connections[:-1]

    train_opt.filter_size /= 2
    x = Unet_decoder_layer(x, train_opt.kernel, train_opt.filter_size, train_opt.pool_size, residual_connections[-1])
    residual_connections = residual_connections[:-1]

    train_opt.filter_size /= 2
    x = Unet_decoder_layer(x, train_opt.kernel, train_opt.filter_size, train_opt.pool_size, residual_connections[-1])

    final_layer = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    return final_layer


input_layer = Input((None, None, 3))
output_layer = Unet(input_layer)

model = Model(inputs=input_layer, outputs=output_layer)

print(model.summary())

plot_model(model, to_file='model.png', show_shapes=True)

# Compile model
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

# Define checkpoint
checkpoint = ModelCheckpoint(filepath='model.hdf5', save_best_only=True, verbose=train_opt.checkpoint_verbose)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_dice_coef', mode='min',
    factor=train_opt.factor, patience=train_opt.reduce_lr_patience, min_lr=train_opt.min_lr,
    verbose=train_opt.reduce_lr_verbose)

# Stop training when a monitored metric has stopped improving
early = EarlyStopping(patience=train_opt.early_patience, verbose=train_opt.early_verbose)

# Define train data generator
training_generator = TrainDataGenerator(
    train_ID, train_labels,
    train_image_dir,
    batch_size=train_opt.batch_size,
    dim=train_opt.image_shape,
    augmentation=augmentor)

# Define valid data generator
validation_generator = TrainDataGenerator(
    valid_ID, valid_labels,
    train_image_dir,
    batch_size=train_opt.batch_size,
    dim=train_opt.image_shape,
    shuffle=False)

history = model.fit(training_generator,
                    steps_per_epoch=train_opt.steps_per_epoch,
                    epochs=train_opt.epochs,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    callbacks=[checkpoint, early, reduce_lr],
                    verbose=1)
