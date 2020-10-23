from wide_resnet import WRN_28_2
from training_loop import *
import tensorflow_datasets as tfds

if __name__ == '__main__':
    # temporary. Should be our own dataset_loader
    ds = tfds.load('cifar10', as_supervised=True)

    wrn_28_2 = WRN_28_2()
    training(wrn_28_2, ds)

