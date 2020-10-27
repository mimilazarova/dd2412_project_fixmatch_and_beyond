from wide_resnet import WRN_28_2
from error import test_error
# import tensorflow_datasets as tfds
from load_data import *
import sys
import logging
from training_loop import training

# hyperparams   (most are from section 4 in the FixMatch paper)
lamda = 1     # proportion of unlabeled loss in total loss
eta = 0.03    # learning rate
beta = 0.09   # momentum
tau = 0.95    # threshold in pseudo-labeling
mu = 0.7      # proportion of unlabeled samples in batch
B = 64        # number of labeled examples in batch (in training)
K = 2 ** 20   # number of training steps in total
nesterov = False
batch_size = 64  # should be 64?
epochs = 50
# weight decay
# SGD instead of Adam


#CTAugment params
cta_classes = 10
cta_decay = 0.99
cta_depth = 2
cta_threshold = 0.8

hparams = {'lamda': lamda, 'eta': eta, 'beta': beta, 'tau': tau, 'mu': mu, 'B': B, 'K': K, 'nesterov': False, 'batch_size': batch_size,
           'epochs': epochs,
           'cta_classes': cta_classes, 'cta_decay': cta_decay, 'cta_depth': cta_depth, 'cta_threshold': cta_threshold}

def main(argv):
    logging.info("now in main")
    data_directory = argv[0]
    dataset = argv[1]
    seed = argv[2]
    n_label = argv[3]
    n_classes = argv[4]
    test_directory = argv[5]
    logging.info("args read")
    lds, uds, labels = LoadAll(data_directory, dataset, seed, n_label)
    logging.info("datasets loaded")
    test, test_labels = LoadTest(test_directory, dataset)
    logging.info("test dataset loaded")

    wrn_28_2 = WRN_28_2()
    logging.info("model created")
    model = training(wrn_28_2, lds, uds, labels, hparams, n_classes)
    logging.info("model trained")
    err = test_error(model, test, test_labels)
    logging.info("Test accuracy: {} on {}.{}@{}-label".format(err, dataset, seed, n_label))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
    logging.info("Done, signing off...")

