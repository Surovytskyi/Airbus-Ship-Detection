import argparse as ap

def train_options():
    parser = ap.ArgumentParser()
    parser.add_argument("--image_shape", default=(256, 256), type=int, help='size of train images')
    parser.add_argument("--batch_size", default=8, type=int, help='size of trining batch')
    parser.add_argument("--kernel", default=3, type=int, help='number of Unet kernel')
    parser.add_argument("--filter_size", default=64, type=int, help='size of Unet filter')
    parser.add_argument("--pool_size", default=2, type=int, help='size of Unet pool')
    parser.add_argument("--steps_per_epoch", default=50, type=int, help='steps per epoch')
    parser.add_argument("--epochs", default=100, type=int, help='number of epochs')
    parser.add_argument("--early_patience", default=10, type=int,
                        help='EarlyStopping: number of epochs with no improvement after which training will be stopped')
    parser.add_argument("--early_verbose", default=1, type=int, help='EarlyStopping: verbose')
    parser.add_argument("--factor", default=0.3, type=float,
                        help='ReduceLROnPlateau: factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument("--reduce_lr_patience", default=3, type=int,
                        help='ReduceLROnPlateau: number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--min_lr", default=0.00001, type=float, help='ReduceLROnPlateau: lower bound on the learning rate')
    parser.add_argument("--reduce_lr_verbose", default=1, type=int, help='ReduceLROnPlateau: verbose')
    parser.add_argument("--checkpoint_verbose", default=1, type=int, help='ModelCheckpoint: verbose')
    train_opt = parser.parse_args()
    return train_opt

def test_options():
    parser = ap.ArgumentParser()
    parser.add_argument("--image_shape", default=(768, 768), type=int, help='size of test images')
    parser.add_argument("--batch_size", default=8, type=int, help='size of test batch')
    parser.add_argument("--threshold", default=0.5, type=float, help='Confusion matrix threshold')
    parser.add_argument("--verbose", default=1, type=int, help='predictions verbose')

    test_opt = parser.parse_args()
    return test_opt

