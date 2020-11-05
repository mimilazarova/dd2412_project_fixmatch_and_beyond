from wide_resnet import WRN_28_2, RN_16
from error import test_error
# import tensorflow_datasets as tfds
from load_data import *
import sys
import logging
#from training_loop import training
from new_training_loop import training

# hyperparams   (most are from section 4 in the FixMatch paper)
lamda = 1     # proportion of unlabeled loss in total loss
eta = 0.03/4  # learning rate
beta = 0.9   # momentum
tau = 0.95    # threshold in pseudo-labeling
mu = 7        # proportion of unlabeled samples in batch
B = 16        # batch size
#K = 58593    # number of training steps in total
nesterov = True
epochs = 150

# Weight decay: CIFAR10 CIFAR100 SVHN   STL-10
#               0.0005  0.001    0.0005 0.0005
weight_decay = 0.0005


# weight decay
# SGD instead of Adam


#CTAugment params
cta_decay = 0.99
cta_depth = 2
cta_threshold = 0.8

hparams = {'lamda': lamda, 'eta': eta, 'beta': beta, 'tau': tau, 'mu': mu, 'B': B, 'nesterov': False,
           'epochs': epochs, 'weight_decay': weight_decay,
           'cta_decay': cta_decay, 'cta_depth': cta_depth, 'cta_threshold': cta_threshold}

def main(argv):
    print(argv)
    logging.info("now in main")
    data_directory = argv[0]
    dataset = argv[1]
    seed = argv[2]
    n_label = argv[3]
    n_classes = int(argv[4])
    hparams['cta_classes'] = n_classes
    test_directory = argv[5]
    logging.info("args read")
    lds, uds, labels = LoadAll(data_directory, dataset, seed, n_label)
    logging.info("datasets loaded")
    test, test_labels = LoadTest(test_directory, dataset)
    logging.info("test dataset loaded")

    K = int(uds.shape[0]/(B*mu))*epochs
    hparams['K'] = K

    rn = WRN_28_2()
    # rn = RN_16()
    logging.info("model created")
    model = training(rn, lds, uds, labels, hparams, n_classes)
    logging.info("model trained")
    err = test_error(model, test, test_labels)
    logging.info("Test accuracy: {} on {}.{}@{}-label".format(err, dataset, seed, n_label))
    print("Test accuracy: {} on {}.{}@{}-label".format(err, dataset, seed, n_label))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
    logging.info("Done, signing off...")

