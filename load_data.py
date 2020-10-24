import tensorflow as tf
import numpy as np

def ParseFunction(serialized, image_shape=[32, 32, 3]):
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}

    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features) 
    image = tf.image.decode_image(parsed_example['image'])
    image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    data = dict(image=image, label=parsed_example['label'])
    return data

def LoadData(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(ParseFunction)
    
    # it = tf.compat.v1.data.make_one_shot_iterator(dataset) # Never used?
    images = np.stack([x['image'] for x in dataset])
    labels = np.stack([x['label'] for x in dataset])

    return images, labels