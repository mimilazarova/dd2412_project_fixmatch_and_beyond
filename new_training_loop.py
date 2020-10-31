import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm, tqdm_notebook
from augment import CTAugment

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import math


class OurCosineDecay(tf.keras.experimental.CosineDecay):

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "CosineDecay"):
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = math_ops.cos(
                constant_op.constant(7 / 16 * math.pi) * completed_fraction)

            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return math_ops.multiply(initial_learning_rate, decayed)


def training(model, full_x_l, full_x_u, full_y_l, hparams, n_classes, mean=None, std=None,
             val_interval=2000, log_interval=200):
    def weak_transformation(x):
        x = tf.image.random_flip_left_right(x)
        max_shift = tf.cast(x.shape[1] * 0.125, dtype=tf.dtypes.int32)
        shift = tf.random.uniform([x.shape[0], 2], minval=-max_shift, maxval=max_shift, dtype=tf.dtypes.int32)
        return tfa.image.translate(x, tf.cast(shift, tf.dtypes.float32))

    def pseudolabel(class_dist):
        argmax = tf.math.argmax(class_dist, axis=1)
        return tf.one_hot(argmax, class_dist.shape[1])

    def threshold_gate(one_hot, logits, threshold):
        max_probs = tf.math.multiply(one_hot, tf.nn.softmax(logits))
        return tf.cast(max_probs > threshold, max_probs.dtype)  # * max_probs

    def sample_labeled_data(ds=full_x_l, y=full_y_l, batch_size=hparams['batch_size']):
        total_samples = ds.shape[0]
        if total_samples >= batch_size:
            choices = np.random.choice(np.arange(total_samples), batch_size, replace=False)
        else:
            choices = np.random.choice(np.arange(total_samples), batch_size, replace=True)

        x_l = ds[choices, :, :, :]
        y_l = y[choices]

        return x_l, y_l

    def step(x_l, y_l, x_u):
        with tf.GradientTape() as tape:

            # labeled data
            x_l_weak = weak_transformation(x_l)
            output_l_weak = model(x_l_weak, True)
            loss_l = loss_fn_l(y_l, output_l_weak)

            # update CTAugment weights
            x_l_strong, choices, bins = cta.augment_batch(x_l)
            output_l_strong = model(x_l_strong, True)
            cta.update_weights_batch(y_l, output_l_strong, choices, bins)

            # unlabeled data
            x_u_weak = weak_transformation(x_u)
            output_u_weak = model(x_u_weak, True)
            y_u = pseudolabel(output_u_weak)
            y_u = threshold_gate(y_u, output_u_weak, hparams['tau'])
            x_u_strong, choices, bins = cta.augment_batch(x_u)
            output_u_strong = model(x_u_strong, True)
            loss_u = loss_fn_u(y_u, output_u_strong)

            # add losses together
            loss = loss_l + hparams['lamda'] * loss_u

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))


    schedule = OurCosineDecay(hparams['eta'], hparams['K'])
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=hparams['beta'], nesterov=hparams['nesterov'])

    loss_fn_u = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_fn_l = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    cta = CTAugment(hparams['cta_classes'], hparams['cta_decay'], hparams['cta_threshold'], hparams['cta_depth'])

    # ds_l = tf.data.Dataset.from_tensor_slices((full_x_l, full_y_l))
    ds_u = tf.data.Dataset.from_tensor_slices(full_x_u)

    # split into batches
    # ds_l = ds_l.batch(hparams['batch_size']).prefetch(-1)
    ds_u = ds_u.batch(int(hparams['mu'] * hparams['batch_size'])).prefetch(-1)
    # if type casting needed: x = tf.cast(x, tf.float32)

    training_step = 0

    # for epoch in range(hparams['epochs']):
    #         for (x_l, y_l), x_u in tqdm(zip(ds_l, ds_u), desc='epoch {}/{}'.format(epoch + 1, hparams['epochs']),
    #                                     total=val_interval, ncols=100, ascii=True):
    #             training_step += 1
    #             step(x_l, y_l, x_u)

    for epoch in range(hparams['epochs']):
            for x_u in tqdm(ds_u, desc='epoch {}/{}'.format(epoch + 1, hparams['epochs']),
                                        total=val_interval, ncols=100, ascii=True):
                training_step += 1
                x_l, y_l = sample_labeled_data()
                step(x_l, y_l, x_u)

    return model