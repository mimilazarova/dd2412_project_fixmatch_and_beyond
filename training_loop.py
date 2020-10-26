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


def training(model, ds_l, ds_u, hparams, n_classes, mean=None, std=None,
             val_interval=2000, log_interval=200):
    def train_prep(x, y):
        x = tf.cast(x, tf.float32)  # /255.
        return x, y

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

    def split_data_into_arrays(ds):
        images = []
        labels = []
        for ex in ds.take(50000000):
            try:
                images.append(ex[0])
                labels.append(ex[1])
            except:
                break
        return np.stack(images), np.stack(labels)

    # @tf.function
    def step(x_l, y_l, x_u, n_classes, training):
        with tf.GradientTape() as tape:
            tf.print("y_l", y_l)

            if y_l.ndim is 1:
                y_l = tf.one_hot(y_l, n_classes)

                # labeled data
            x_l_weak = weak_transformation(x_l)
            output_l = model(x_l_weak, training)

            loss_l = loss_fn(y_l, output_l)

            # unlabeled data
            x_u_weak = weak_transformation(x_u)
            output_u_weak = model(x_u_weak, training)  # should this be training or not?
            y_u = pseudolabel(output_u_weak)
            y_u = threshold_gate(y_u, output_u_weak, hparams['tau'])

            x_u_strong, choices, bins = cta.augment_batch(x_u)
            output_u_strong = model(x_u_strong, training)
            cta.update_weights_batch(y_u, output_u_strong, choices, bins)  #

            loss_u = loss_fn(y_u, output_u_strong)

            # tf.print(loss_u)
            # print(loss_u)

            # add losses together
            loss = loss_l + hparams['lamda'] * loss_u

        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        class_pred = tf.math.argmax(output_l, axis=1)
        tmp = tf.argmax(y_l, axis=1)
        # accuracy(tmp, class_pred)
        labeled_loss(loss_l)
        unlabeled_loss(loss_u)

        labels = np.argmax(y_u, axis=1)
        labels[np.sum(y_u, axis=1) == 0] = -1
        return labels

    schedule = OurCosineDecay(hparams['eta'], hparams['K'])
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=hparams['beta'], nesterov=hparams['nesterov'])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    cta = CTAugment(hparams['cta_classes'], hparams['cta_decay'], hparams['cta_threshold'], hparams['cta_depth'])

    full_x_l, full_y_l = split_data_into_arrays(ds_l)
    full_x_u, _ = split_data_into_arrays(ds_u)

    # split into batches
    ds_l = ds_l.map(train_prep).batch(hparams['batch_size']).prefetch(-1)
    ds_u = ds_u.map(train_prep).batch(hparams['batch_size']).prefetch(-1)

    # runid = run_name + '_x' + str(np.random.randint(10000))
    # writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    labeled_loss = tf.metrics.Mean()
    unlabeled_loss = tf.metrics.Mean()

    # print(f"RUNID: {runid}")
    # tf.keras.utils.plot_model(model)#, os.path.join('saved_plots', runid + '.png'))

    training_step = 0
    best_validation_acc = 0
    epochs = hparams['epochs']
    for epoch in range(epochs):

        y_u = np.array([])

        for (x_l, y_l), x_u in tqdm(zip(ds_l, ds_u), #desc=f'epoch {epoch + 1}/{epochs}',
                                         total=val_interval, ncols=100, ascii=True):
            tf.print("step")

            training_step += 1
            y_batch = step(x_l, y_l, x_u, n_classes, training=True)
            # y_batch[1] = np.random.randint(0, 9)
            y_u = np.concatenate((y_u, y_batch))

            # tf.print(y_batch)

            if training_step % log_interval == 0:
                # with writer.as_default():
                loss_l, loss_u, err = labeled_loss.result(), unlabeled_loss.result(), 1 - accuracy.result()
                print(f" loss_l: {loss_l:^6.3f} | loss_u: {loss_u:^6.3f} | err: {err:^6.3f}", end='\r')

                tf.summary.scalar('train/error_rate', err, training_step)
                tf.summary.scalar('train/labeled_loss', loss_l, training_step)
                tf.summary.scalar('train/unlabeled_loss', loss_u, training_step)
                tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                labeled_loss.reset_states()
                unlabeled_loss.reset_states()
                accuracy.reset_states()

        tf.print(full_x_u.shape, full_x_l.shape)
        y_dim = y_u.shape[0]
        tf.print(y_u.shape)
        tf.print("number unlabeled samlples {}".format(y_dim))

        # Update labeled and unlabeled datasets
        tf.print(full_x_u.shape, y_dim, y_u.shape)
        new_x_l = [full_x_u[i, :, :, :] for i in range(y_dim) if y_u[i] > -1]
        new_y_l = [y_u[i] for i in range(y_dim) if y_u[i] > -1]

        if len(new_x_l) > 0:
            new_x_l = np.stack(new_x_l)
            new_y_l = np.stack(new_y_l)

            new_x_u = [full_x_u[i, :, :, :] for i in range(y_dim) if y_u[i] == -1]
            full_x_u = new_x_u

            if len(full_x_u) > 0:
                full_x_u = np.stack(full_x_u)

            full_x_l = np.concatenate((full_x_l, new_x_l))
            full_y_l = np.concatenate((full_y_l, new_y_l)).astype(np.int64)

            # print(full_x_l.shape, full_y_l.shape, full_x_u.shape, np.ones([full_x_u.shape[0],]).shape)
            ds_l = tf.data.Dataset.from_tensor_slices((full_x_l, full_y_l))
            ds_u = tf.data.Dataset.from_tensor_slices(full_x_u)

            # ds_u.apply(tf.data.experimental.unbatch())

            ds_l = ds_l.map(train_prep).batch(hparams['batch_size']).prefetch(-1)
            ds_u = ds_u.map(train_prep).batch(hparams['batch_size']).prefetch(-1)

        labeled_loss.reset_states()
        unlabeled_loss.reset_states()
        accuracy.reset_states()