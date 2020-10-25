# from wide_resnet import WRN_28_2
# from training_loop import *
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
    logging.info("args read")
    lds, uds, lables = load_all(*argv)
    logging.info("datasets loaded")

    # wrn_28_2 = WRN_28_2()
    # training(wrn_28_2, ds)


if __name__ == "__main__":
   print(sys.argv)
   main(sys.argv[1:])
   print("done")

