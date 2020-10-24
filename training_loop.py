
#import pip
#import sys
#import subprocess
#!pip install tensorflow-addons==0.11.1
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-addons==0.11.1'])

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tqdm import tqdm, tqdm_notebook

from augment import CTAugment


def training(model, ds, mean=None, std=None, lr_values=[0.01, 0.1, 0.01, 0.001],
             lr_boundaries=[400, 32000, 48000, 64000],
             val_interval=2000, log_interval=200, batch_size=128, nesterov=False):
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries[:-1], values=lr_values)
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9, nesterov=nesterov)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train_prep(x, y):
        x = tf.cast(x, tf.float32) / 255.
        # x = tf.image.random_flip_left_right(x)
        # x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40) #must fix ds independent shape
        # x = tf.image.random_crop(x, (32, 32, 3))          #must fix ds independent shape
        if mean is not None and std is not None:
            x = (x - mean) / std
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32) / 255.
        if mean is not None and std is not None:
            x = (x - mean) / std
        return x, y

    # ds['train'] = ds['train'].map(train_prep).shuffle(10000).repeat().batch(batch_size).prefetch(-1)
    ds['train'] = ds['train'].map(train_prep).batch(batch_size).prefetch(-1)
    ds['test'] = ds['test'].map(valid_prep).batch(batch_size * 4).prefetch(-1)

    # runid = run_name + '_x' + str(np.random.randint(10000))
    # writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    reg_loss = tf.metrics.Mean()

    # print(f"RUNID: {runid}")
    # tf.keras.utils.plot_model(model)#, os.path.join('saved_plots', runid + '.png'))

    def weak_transformation(x):
        x = tf.image.random_flip_left_right(x)
        max_shift = tf.cast(x.shape[1] * 0.125, dtype=tf.dtypes.int32)
        shift = tf.random.uniform([x.shape[0], 2], minval=-max_shift, maxval=max_shift, dtype=tf.dtypes.int32)
        return tfa.image.translate(x, tf.cast(shift, tf.dtypes.float32))

    def strong_transformation(x):
        cta = CTAugment(10, 0.99, 0.8, 2, 10)
        return cta.augment(x)

    def pseudolabel(class_dist):
        argmax = tf.math.argmax(class_dist, axis=1)
        return tf.one_hot(argmax, class_dist.shape[1])

    def threshold_gate(one_hot, logits, threshold):
        max_probs = tf.math.multiply(one_hot, tf.nn.softmax(logits))
        return tf.cast(max_probs > threshold, max_probs.dtype)  # * max_probs

    # @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            # unlabeled data
            x_wk = weak_transformation(x)
            outs_wk = model(x_wk, training)  # should this be training or not?
            weak_labels = pseudolabel(outs_wk)
            weak_labels = threshold_gate(weak_labels, outs_wk, 0.7)

            cta = CTAugment(10, 0.99, 0.8, 2)

            # x_str = cta.augment(x)
            # x_str = cta.augment_batch(x)

            x_str, choices, bins = cta.augment_batch(x)

            outs_str = model(x_str, training)

            cta.update_weights_batch(weak_labels, outs_str, choices, bins)

            # unlabeled_loss = loss_fn(weak_labels, outs_str)

            # labeled data
            # outs = model(x, training)
            # labeled_loss = loss_fn(y, outs)

            # add losses together
            # loss = labeled_loss + lambda * unlabeled_loss

            r_loss = tf.add_n(model.losses)
            outs = model(x, training)
            c_loss = loss_fn(y, outs)
            loss = c_loss + r_loss

        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        accuracy(y, outs)
        cls_loss(c_loss)
        reg_loss(r_loss)

    training_step = 0
    best_validation_acc = 0
    epochs = lr_boundaries[-1] // val_interval

    for epoch in range(epochs):
        for x, y in tqdm(ds['train'].take(val_interval), desc=f'epoch {epoch + 1}/{epochs}',
                         total=val_interval, ncols=100, ascii=True):

            training_step += 1
            step(x, y, training=True)

            if training_step % log_interval == 0:
                # with writer.as_default():
                c_loss, r_loss, err = cls_loss.result(), reg_loss.result(), 1 - accuracy.result()
                print(f" c_loss: {c_loss:^6.3f} | r_loss: {r_loss:^6.3f} | err: {err:^6.3f}", end='\r')

                tf.summary.scalar('train/error_rate', err, training_step)
                tf.summary.scalar('train/classification_loss', c_loss, training_step)
                tf.summary.scalar('train/regularization_loss', r_loss, training_step)
                tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                cls_loss.reset_states()
                reg_loss.reset_states()
                accuracy.reset_states()

        for x, y in ds['test']:
            step(x, y, training=False)

        # with writer.as_default(): TBULATE THE FOLLOWING WHEN UNCOMMENTING!
        tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
        tf.summary.scalar('test/error_rate', 1 - accuracy.result(), step=training_step)

        if accuracy.result() > best_validation_acc:
            best_validation_acc = accuracy.result()
            # model.save_weights(os.path.join('saved_models', runid + '.tf'))

        cls_loss.reset_states()
        accuracy.reset_states()


def cifar_error_test(model, tr_len=20, vd_len=2):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(0.01)

    ds = tfds.load('cifar10', as_supervised=True, in_memory=True)
    std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
    mean = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))

    def train_prep(x, y):
        x = tf.cast(x, tf.float32) / 255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32) / 255.
        x = (x - mean) / std
        return x, y

    ds['train'] = ds['train'].map(train_prep).batch(5).take(tr_len).prefetch(-1)
    ds['test'] = ds['test'].map(valid_prep).batch(5).take(vd_len).prefetch(-1)

    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    reg_loss = tf.metrics.Mean()

    # @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            r_loss = tf.add_n(model.losses)
            outs = model(x, training)
            c_loss = loss_fn(y, outs)
            loss = c_loss + r_loss

        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        accuracy(y, outs)
        cls_loss(c_loss)
        reg_loss(r_loss)

    training_step = 0
    for x, y in tqdm(ds['train'], desc=f'test', total=tr_len, ncols=100, ascii=True):
        training_step += 1
        step(x, y, training=True)
        c_loss, r_loss, err = cls_loss.result(), reg_loss.result(), 1 - accuracy.result()
        print(f" c_loss: {c_loss:^6.3f} | r_loss: {r_loss:^6.3f} | err: {err:^6.3f}", end='\r')

    for x, y in ds['test']:
        step(x, y, training=False)