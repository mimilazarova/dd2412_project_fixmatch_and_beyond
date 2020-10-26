import tensorflow as tf
import numpy as np

def test_error(model, test_data, test_labels):
    out = model(test_data)
    out_l = np.argmax(out, axis=1)
    return np.sum(out_l == test_labels)/len(test_labels)
