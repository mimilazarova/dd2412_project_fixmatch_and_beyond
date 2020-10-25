import tensorflow as tf
import numpy as np
import json
import os


def ParseFunction(serialized, image_shape=[32, 32, 3]):
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}

    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features) 
    image = tf.image.decode_image(parsed_example['image'])
    image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    data = dict(image=image, label=parsed_example['label'])
    return data


def LoadData(filename, tensor=False):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(ParseFunction)
    
    # it = tf.compat.v1.data.make_one_shot_iterator(dataset) # Never used?
    images = np.stack([x['image'] for x in dataset])
    labels = np.stack([x['label'] for x in dataset])

    if tensor:
        return tf.data.Dataset.from_tensor_slices((images, labels))
    else:
        return images, labels
    

def load_all(dir, dataset, seed, n_labeled):
    l_data_fname = os.path.join(dir, "{}.{}@{}-label.tfrecord".format(dataset, str(seed), str(n_labeled)))
    l_json_fname = os.path.join(dir, "{}.{}@{}-label.json".format(dataset, str(seed), str(n_labeled)))
    u_data_fname = os.path.join(dir, "{}-unlabel.tfrecord".format(dataset))
    u_json_fname = os.path.join(dir, "{}-unlabel.json".format(dataset))

    with open(l_json_fname, "r") as f:
        l_json = json.load(f)['label']


    with open(u_json_fname, "r") as f:
        u_json = json.load(f)['indexes']

    ds_l, ls = LoadData(l_data_fname)
    ds_u, _ = LoadData(u_data_fname)

    new_ds_u = np.stack([ds_u[i, :, :, :] for i in u_json if i not in l_json])

    return ds_l, new_ds_u, ls


def load_test(dir, dataset):
    data_fname = os.path.join(dir, "{}-test.tfrecord".format(dataset))

    return LoadData(data_fname)


