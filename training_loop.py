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


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def shuffle(a):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
    return shuffled_a


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


    # @tf.function
    def step(x_l, y_l, x_u, training):
        with tf.GradientTape() as tape:

            # labeled data
            x_l_weak = weak_transformation(x_l)
            output_l_weak = model(x_l_weak, training)
            loss_l = loss_fn_l(y_l, output_l_weak)

            # update CTAugment weights
            x_l_strong, choices, bins = cta.augment_batch(x_l)
            output_l_strong = model(x_l_strong, training)
            cta.update_weights_batch(y_l, output_l_strong, choices, bins)

            # unlabeled data
            x_u_weak = weak_transformation(x_u)
            output_u_weak = model(x_u_weak, training)  # should this be training or not?
            y_u = pseudolabel(output_u_weak)
            y_u = threshold_gate(y_u, output_u_weak, hparams['tau'])
            x_u_strong, choices, bins = cta.augment_batch(x_u)
            output_u_strong = model(x_u_strong, training)
            loss_u = loss_fn_u(y_u, output_u_strong)


            # # labeled data
            # x_l_weak = weak_transformation(x_l)
            # output_l = model(x_l_weak, training)
            #
            # loss_l = loss_fn_l(y_l, output_l)
            #
            # # unlabeled data
            # x_u_weak = weak_transformation(x_u)
            # output_u_weak = model(x_u_weak, training)  # should this be training or not?
            # y_u = pseudolabel(output_u_weak)
            # y_u = threshold_gate(y_u, output_u_weak, hparams['tau'])
            #
            # x_u_strong, choices, bins = cta.augment_batch(x_u)
            # output_u_strong = model(x_u_strong, training)
            # cta.update_weights_batch(y_u, output_u_strong, choices, bins)  #
            #
            # loss_u = loss_fn_u(y_u, output_u_strong)

            # add losses together
            loss = loss_l + hparams['lamda'] * loss_u

        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        labeled_loss(loss_l)
        unlabeled_loss(loss_u)

        labels = np.argmax(y_u, axis=1)
        labels[np.sum(y_u, axis=1) == 0] = -1
        return labels

    schedule = OurCosineDecay(hparams['eta'], hparams['K'])
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=hparams['beta'], nesterov=hparams['nesterov'])
    loss_fn_u = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_fn_l = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    cta = CTAugment(hparams['cta_classes'], hparams['cta_decay'], hparams['cta_threshold'], hparams['cta_depth'])

    ds_l = tf.data.Dataset.from_tensor_slices((full_x_l, full_y_l))
    ds_u = tf.data.Dataset.from_tensor_slices(full_x_u)

    # split into batches
    ds_l = ds_l.batch(hparams['batch_size']).prefetch(-1)#.map(train_prep).batch(hparams['batch_size']).prefetch(-1)
    ds_u = ds_u.batch(hparams['batch_size']).prefetch(-1)#.map(unlabelled_prep).batch(hparams['batch_size']).prefetch(-1)
    # if type casting needed: x = tf.cast(x, tf.float32)

    # runid = run_name + '_x' + str(np.random.randint(10000))
    # writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    labeled_loss = tf.metrics.Mean()
    unlabeled_loss = tf.metrics.Mean()

    # print(f"RUNID: {runid}")
    # tf.keras.utils.plot_model(model)#, os.path.join('saved_plots', runid + '.png'))

    training_step = 0
    epochs = hparams['epochs']
    for epoch in range(epochs):

        y_u = np.array([])
        for (x_l, y_l), x_u in tqdm(zip(ds_l, ds_u), desc='epoch {}/{}'.format(epoch+1, epochs), total=val_interval, ncols=100, ascii=True):

            training_step += 1
            y_batch = step(x_l, y_l, x_u, training=True)
            y_u = np.concatenate((y_u, y_batch), axis=None)

            if training_step % log_interval == 0:
                # with writer.as_default():
                loss_l, loss_u, err = labeled_loss.result(), unlabeled_loss.result(), 1 - accuracy.result()
                #print(f" loss_l: {loss_l:^6.3f} | loss_u: {loss_u:^6.3f} | err: {err:^6.3f}", end='\r')

                tf.summary.scalar('train/error_rate', err, training_step)
                tf.summary.scalar('train/labeled_loss', loss_l, training_step)
                tf.summary.scalar('train/unlabeled_loss', loss_u, training_step)
                tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                labeled_loss.reset_states()
                unlabeled_loss.reset_states()
                accuracy.reset_states()

        tf.print(full_x_u.shape, full_x_l.shape, y_u.shape, full_y_l.shape)
        y_dim = y_u.shape[0]
        all_y_dim = full_x_u.shape[0]
        # Update labeled and unlabeled datasets
        new_x_l = [full_x_u[i, :, :, :] for i in range(y_dim) if y_u[i] > -1]
        new_y_l = [y_u[i] for i in range(y_dim) if y_u[i] > -1]

        if len(new_x_l) > 0:
            new_x_l = np.stack(new_x_l)
            new_y_l = np.stack(new_y_l)

            new_x_u = [full_x_u[i, :, :, :] for i in range(all_y_dim) if i >= y_dim or y_u[i] == -1]
            full_x_u = new_x_u

            if len(full_x_u) > 0:
                full_x_u = np.stack(full_x_u)

            full_x_l = np.concatenate((full_x_l, new_x_l))
            full_y_l = np.concatenate((full_y_l, new_y_l), axis=None).astype(np.int64)
            tf.print(full_x_u.shape, full_x_l.shape, y_u.shape, full_y_l.shape)

            full_x_l, full_y_l = shuffle_in_unison(full_x_l, full_y_l)
            full_x_u = shuffle(full_x_u)

            ds_l = tf.data.Dataset.from_tensor_slices((full_x_l, full_y_l))
            ds_u = tf.data.Dataset.from_tensor_slices(full_x_u)

            ds_l = ds_l.batch(hparams['batch_size']).prefetch(-1)#.map(train_prep).batch(hparams['batch_size']).prefetch(-1)
            ds_u = ds_u.batch(hparams['batch_size']).prefetch(-1)#.map(unlabelled_prep).batch(hparams['batch_size']).prefetch(-1)

        labeled_loss.reset_states()
        unlabeled_loss.reset_states()
        accuracy.reset_states()
