from wide_resnet import WRN_28_2
from training_loop import test_error
# import tensorflow_datasets as tfds
from load_data import *
import sys
import logging


def main(argv):
    logging.info("now in main")
    data_directory = argv[0]
    dataset = argv[1]
    seed = argv[2]
    n_label = argv[3]
    test_directory = argv[4]
    logging.info("args read")
    lds, uds, lables = LoadAll(data_directory, dataset, seed, n_label)
    logging.info("datasets loaded")
    test, test_labels = LoadTest(test_directory, dataset)
    logging.info("test dataset loaded")

    wrn_28_2 = WRN_28_2()
    logging.info("model created")
    # training(wrn_28_2, ds)

    err = test_error(wrn_28_2, test, test_labels)
    logging.info("{} on {}.{}@{}-label".format(err, dataset, seed, n_label))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
    logging.info("Done, signing off...")

