import os
import pandas as pd
import numpy as np

from sklearn import metrics

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as K
import tensorflow.keras.layers as layers

# from catalyst.data.sampler import BalanceClassSampler
# from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass
# import pytorch_lightning as pl

from datasets import load_dataset, Hardload_Generator, get_test_augmentations, get_train_augmentations
from models.scan import SCAN
from loss import TripletLoss
from metrics import eval_from_scores
from utils import GridMaker, read_label

strategy = tf.distribute.MirroredStrategy()

class LightningModel(K.Model):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = SCAN()
        self.triplet_loss = TripletLoss()
        self.optimizer = self.configure_optimizers()
        self.log_cues = not self.hparams.cue_log_every == 0
        # self.grid_maker = GridMaker()
        if self.hparams.use_focal_loss:
            self.clf_criterion = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        else:
            self.clf_criterion = K.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, x, training=False):
        return self.model(x, training)

    def infer(self, x):
        outs, clf_out = self.model(x, training=False)
        return tf.nn.softmax(clf_out)[:,1]

    def calc_losses(self, outs, clf_out, target):

        clf_loss = (
            self.clf_criterion(y_true=target, y_pred=clf_out)
            * self.hparams.loss_coef["clf_loss"]
        )
        cue = outs[-1]
        target_mask = tf.where(tf.equal(1,tf.cast(target,tf.int32)),tf.zeros_like(target),tf.ones_like(target))
        target_mask = tf.cast(tf.reshape(target_mask, [-1, 1, 1, 1]), tf.float32)
        cue *= target_mask
        num_reg = tf.math.reduce_sum(tf.cast(tf.equal(0,tf.cast(target,tf.int32)), tf.float32)) \
                * tf.cast(cue.shape[1] * cue.shape[2] * cue.shape[3], tf.float32)
        reg_loss = (
            tf.math.reduce_sum(tf.math.abs(cue)) / (num_reg + 1e-9)
        ) * self.hparams.loss_coef["reg_loss"]

        trip_loss = 0
        for feat in outs[:-1]:
            feat = layers.GlobalAveragePooling2D()(feat)
            trip_loss += (
                self.triplet_loss(feat, target)
                * self.hparams.loss_coef["trip_loss"]
            )
        total_loss = clf_loss + reg_loss + trip_loss

        return total_loss, clf_loss, reg_loss, trip_loss

    with strategy.scope():
        @tf.function
        def training_step(self, batch):
            input_ = batch[0]
            target = batch[1]
            with tf.GradientTape() as tape:
                outs, clf_out = self(input_, training=True)
                loss, clf_loss, reg_loss, trip_loss = self.calc_losses(outs, clf_out, target)
            gradient = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

            tensorboard_logs = {
                "train_loss": loss,
                "clf_loss": clf_loss,
                "reg_loss": reg_loss,
                "trip_loss": trip_loss,
                }
            return {"loss": loss, "log": tensorboard_logs}

        def training_epoch_end(self, outputs: list):
            avg_loss = tf.math.reduce_mean(tf.concat([x["loss"] for x in outputs], axis=0))
            clf_loss = tf.math.reduce_mean(tf.concat([x['log']["clf_loss"] for x in outputs], axis=0))
            reg_loss = tf.math.reduce_mean(tf.concat([x['log']["reg_loss"] for x in outputs], axis=0))
            trip_loss = tf.math.reduce_mean(tf.concat([x['log']["trip_loss"] for x in outputs], axis=0))
            tensorboard_logs = {
                "train_avg_loss": avg_loss,
                "train_avg_clf_loss": clf_loss,
                "train_avg_reg_loss": reg_loss,
                "train_avg_trip_loss": trip_loss,
            }
            return {"train_avg_loss": avg_loss, "log": tensorboard_logs}

        # @tf.function
        def validation_step(self, batch):
            input_ = batch[0]
            target = batch[1]
            outs, clf_out = self(input_, training=False)
            loss, *_ = self.calc_losses(outs, clf_out, target)

            loss = tf.math.reduce_mean(loss).numpy()
            metrics_, best_thr, acc = eval_from_scores(clf_out[:,1].numpy(), target.numpy())
            acer, apcer, npcer = metrics_

            return loss, acc, acer

        def validation_epoch_end(self, outputs):
            avg_loss = tf.math.reduce_mean(tf.concat([x["val_loss"] for x in outputs], axis=0))
            targets = np.hstack([output["target"] for output in outputs])
            scores = np.vstack([output["score"] for output in outputs])[:, 1]
            metrics_, best_thr, acc = eval_from_scores(scores, targets)
            acer, apcer, npcer = metrics_
            roc_auc = metrics.roc_auc_score(targets, scores)
            tensorboard_logs = {
                "val_loss": avg_loss,
                "val_roc_auc": roc_auc,
                "val_acer": acer,
                "val_apcer": apcer,
                "val_npcer": npcer,
                "val_acc": acc,
                "val_thr": best_thr,
            }
            return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        scheduler = K.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.hparams.lr
            , decay_steps=5000#self.hparams.milestones
            , decay_rate=0.95#self.hparams.gamma
            , staircase=True
        )
        # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=scheduler, warmup_proportion=0.1, min_lr=1e-6, weight_decay=0.01)
        # optimizer = tfa.optimizers.MovingAverage(optimizer)
        optimizer = K.optimizers.Adam(learning_rate=scheduler)
        return optimizer

    def train_dataloader(self):
        transforms = get_train_augmentations#(self.hparams.image_size)
        tang_files = [[(self.hparams.train_root_tang+sub_dir+'/'+x, read_label(x, anti_word='anti'))
                                           for x in os.listdir(self.hparams.train_root_tang+sub_dir)]
                                       for sub_dir in os.listdir(self.hparams.train_root_tang)
                                       if ('head' not in sub_dir and 'print_attack' not in sub_dir)]
        print_files = [(self.hparams.train_root_tang+sub_dir+'/'+x, read_label(x, anti_word='anti'))
                                           for sub_dir in os.listdir(self.hparams.train_root_tang)
                                           if 'print_attack' in sub_dir
                                       for x in os.listdir(self.hparams.train_root_tang+sub_dir)] \
                    + [(self.hparams.train_root_kp+sub_dir+'/'+x, read_label(x))
                                           for sub_dir in os.listdir(self.hparams.train_root_kp)
                                           if sub_dir.split('_')[-1]==1
                                       for x in os.listdir(self.hparams.train_root_kp+sub_dir)
                                       if 'png' in x]
        dataset = Hardload_Generator(
            tang_files, print_files*20, transforms, self.hparams.batch_size
        )
        '''
        tang_files = [self.hparams.train_root_tang+sub_dir+'/'+x
                                           for sub_dir in os.listdir(self.hparams.train_root_tang)
                                           for x in os.listdir(self.hparams.train_root_tang+sub_dir)]
        tang_dataset = load_dataset(
            tang_files, transforms, anti_word='anti'
        )
        oulu_files = [self.hparams.train_root_oulu+x
                                           for x in os.listdir(self.hparams.train_root_oulu)]
        oulu_dataset = load_dataset(
            oulu_files, transforms,
        )
        dataset = tang_dataset.concatenate(oulu_dataset)
        dataset = strategy.experimnetal_distribute_dataset(tang_dataset)

        dataset = dataset.shuffle(buffer_size = len(tang_files)+len(oulu_files)).cache()
        dataset = dataset.batch(batch_size = self.hparams.batch_size, drop_remainder=False)
        '''

        return dataset

    def val_dataloader(self):
        transforms = get_test_augmentations
        # tang_files = [self.hparams.val_root_tang+x for x in os.listdir(self.hparams.val_root_tang)]
        # tang_dataset = load_dataset(
        #     tang_files, transforms, anti_word='spoof'#, anti_word='fake'
        # )
        # oulu_files = [self.hparams.val_root_oulu+x for x in os.listdir(self.hparams.val_root_oulu)]
        # dataset = load_dataset(
        #     oulu_files, transforms#, anti_word='fake'
        # )
        # dataset = tang_dataset.concatenate(oulu_dataset)
        kp_files = [self.hparams.val_root_kp+x for x in os.listdir(self.hparams.val_root_kp) if 'png' in x]
        dataset = load_dataset(
            kp_files, transforms, real_word='real'
        )

        # dataset = dataset.shuffle(buffer_size = len(tang_files)+len(oulu_files)).cache()
        dataset = dataset.shuffle(buffer_size = len(kp_files)).cache()
        dataset = dataset.batch(batch_size = self.hparams.batch_size, drop_remainder=False)
        # dataset = strategy.experimnetal_distribute_dataset(dataset)

        return dataset
