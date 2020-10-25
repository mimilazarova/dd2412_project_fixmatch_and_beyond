"""
Currently some settings related to original cifar-10-network
that we could probably remove. I've only run this on cifar-10
so I don't know if it will crash or something for other datasets.
// Johan
"""

import os
import tensorflow as tf


def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, padding='same', kernel_regularizer=_regularizer,
                                  kernel_initializer='he_normal', use_bias=False, **kwargs)


def BN_ReLU(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    elif mode == 'B':
        return regularized_padded_conv(filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(filters, 1, strides=stride)(x)
        return tf.keras.layers.BatchNormalization()(x)
    elif mode == 'A':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride > 1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    else:
        raise KeyError("Parameter shortcut_type not recognized!")


def original_block(x, filters, stride=1, **kwargs):
    c1 = regularized_padded_conv(filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, 3)(BN_ReLU(c1))
    c2 = tf.keras.layers.BatchNormalization()(c2)

    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode)
    return tf.keras.layers.ReLU()(x + c2)


def preactivation_block(x, filters, stride=1, preact_block=False):
    flow = BN_ReLU(x)
    if preact_block:
        x = flow

    c1 = regularized_padded_conv(filters, 3, strides=stride)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)

    c2 = regularized_padded_conv(filters, 3)(BN_ReLU(c1))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c2


def bootleneck_block(x, filters, stride=1, preact_block=False):
    flow = BN_ReLU(x)
    if preact_block:
        x = flow

    c1 = regularized_padded_conv(filters // _bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(filters // _bootleneck_width, 3, strides=stride)(BN_ReLU(c1))
    c3 = regularized_padded_conv(filters, 1)(BN_ReLU(c2))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c3


def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
    global _preact_shortcuts
    preact_block = True if _preact_shortcuts or block_idx == 0 else False

    x = block_type(x, filters, stride, preact_block=preact_block)
    for i in range(num_blocks - 1):
        x = block_type(x, filters)
    return x


def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
           shortcut_type='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
           dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True):
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
    _bootleneck_width = bootleneck_width  # used in ResNeXts and bootleneck blocks
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type  # used in blocks
    _cardinality = cardinality  # used in ResNeXts
    _dropout = dropout  # used in Wide ResNets
    _preact_shortcuts = preact_shortcuts

    block_types = {'preactivated': preactivation_block,
                   'bootleneck': bootleneck_block,
                   'original': original_block}

    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)

    if block_type == 'original':
        flow = BN_ReLU(flow)

    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        flow = group_of_blocks(flow,
                               block_type=selected_block,
                               num_blocks=group_size,
                               block_idx=block_idx,
                               filters=feature,
                               stride=stride)

    if block_type != 'original':
        flow = BN_ReLU(flow)

    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_weights_func(model, model_name):
    try:
        model.load_weights(os.path.join('saved_models', model_name + '.tf'))
    except tf.errors.NotFoundError:
        print("No weights found for this model!")
    return model


def cifar_wide_resnet(N, K, block_type='preactivated', shortcut_type='B', dropout=0, l2_reg=2.5e-4):
    assert (N - 4) % 6 == 0, "N-4 has to be divisible by 6"
    lpb = (N - 4) // 6  # layers per block - since N is total number of convolutional layers in Wide ResNet
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(lpb, lpb, lpb),
                   features=(16 * K, 32 * K, 64 * K),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, dropout=dropout, preact_shortcuts=True)
    return model


def WRN_28_2(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4):
    model = cifar_wide_resnet(28, 2, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_28_10')
    return model
