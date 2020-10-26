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


def training(model, ds_l, ds_u, hparams, mean=None, std=None,
             val_interval=2000, log_interval=200, batch_size=128):
    
    schedule = OurCosineDecay(hparams['eta'], hparams['K'])
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=hparams['beta'], nesterov=hparams['nesterov'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    cta = CTAugment(hparams['cta_classes'], hparams['cta_decay'], hparams['cta_threshold'], hparams['cta_depth'])

    def train_prep(x, y):
        x = tf.cast(x, tf.float32) / 255.
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
    ds_l['train'] = ds_l['train'].map(train_prep).batch(hparams['B']).prefetch(-1)
    ds_u['train'] = ds_u['train'].map(train_prep).batch(hparams['mu'] * hparams['B']).prefetch(-1)

    ds_l['test'] = ds_l['test'].map(valid_prep).batch(batch_size * 4).prefetch(-1)

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
            weak_labels = threshold_gate(weak_labels, outs_wk, hparams['treshold'])

            x_str, choices, bins = cta.augment_batch(x)
            outs_str = model(x_str, training)
            cta.update_weights_batch(weak_labels, outs_str, choices, bins)

            unlabeled_loss = loss_fn(weak_labels, outs_str)

            # labeled data
            outs = model(x, training)
            labeled_loss = loss_fn(y, outs)

            # add losses together
            loss = labeled_loss + hparams['lamda'] * unlabeled_loss

            # r_loss = tf.add_n(model.losses)
            # outs = model(x, training)
            # c_loss = loss_fn(y, outs)
            # loss = c_loss + r_loss

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


def test_error(model, test_data, test_labels):
    out = model(test_data)
    out_l = tf.math.argmax(out, axis=1)
    return np.sum(out_l == test_labels)/len(test_labels)
