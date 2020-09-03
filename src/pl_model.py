import os
import pandas as pd
import numpy as np

from sklearn import metrics

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers

# from catalyst.data.sampler import BalanceClassSampler
# from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass
# import pytorch_lightning as pl

from datasets import load_dataset, get_test_augmentations, get_train_augmentations
from models.scan import SCAN
from loss import TripletLoss
from metrics import eval_from_scores
from utils import GridMaker


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
            self.clf_criterion = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        else:
            self.clf_criterion = K.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, x, training=False):
        return self.model(x, training)

    def infer(self, x):
        outs, _ = self.model(x, training=False)
        return outs[-1]

    def calc_losses(self, outs, clf_out, target):

        clf_loss = (
            self.clf_criterion(y_true=target, y_pred=clf_out)
            * self.hparams.loss_coef["clf_loss"]
        )
        cue = outs[-1]
        target_01 = tf.where(tf.equal(1,target),tf.zeros_like(target),tf.ones_like(target))
        target_01 = tf.cast(tf.reshape(target, [-1, 1, 1, 1]), tf.float32)
        cue *= target_01
        num_reg = tf.math.reduce_sum(target_01) \
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

        return total_loss

    @tf.function
    def training_step(self, batch):
        input_ = batch[0]
        target = batch[1]
        with tf.GradientTape() as tape:
            outs, clf_out = self(input_, training=True)
            loss = self.calc_losses(outs, clf_out, target)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = tf.reduce_mean(tf.stack([x["loss"] for x in outputs]))
        tensorboard_logs = {
            "train_avg_loss": avg_loss,
        }
        return {"train_avg_loss": avg_loss, "log": tensorboard_logs}

    @tf.function
    def validation_step(self, batch):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)
        val_dict = {
            "val_loss": loss,
            "score": tf.identity(clf_out),
            "target": tf.identity(target),
        }

        return val_dict

    def validation_epoch_end(self, outputs):
        avg_loss = tf.reduce_mean(tf.stack([x["val_loss"] for x in outputs]))
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
            , decay_steps=1000#self.hparams.milestones
            , decay_rate=0.95#self.hparams.gamma
            , staircase=True
        )
        optim = K.optimizers.Adam(learning_rate=scheduler)
        return optim

    def train_dataloader(self):
        transforms = get_train_augmentations#(self.hparams.image_size)
        tang_files = [self.hparams.train_root_tang+sub_dir+'/'+x
                                           for sub_dir in os.listdir(self.hparams.train_root_tang)
                                           for x in os.listdir(self.hparams.train_root_tang+sub_dir)]
        oulu_files = [self.hparams.train_root_oulu+x
                                           for x in os.listdir(self.hparams.train_root_oulu)]
        tang_dataset = load_dataset(
            tang_files, transforms, anti_word='anti'#, anti_word='fake'
        )
        oulu_dataset = load_dataset(
            oulu_files, transforms, #, anti_word='anti'#, anti_word='fake'
        )
        dataset = tang_dataset.concatenate(oulu_dataset)

        dataset = dataset.shuffle(buffer_size = 1024)
        dataset = dataset.batch(batch_size = self.hparams.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#.cache()

        return dataset

    def val_dataloader(self):
        transforms = get_test_augmentations#(self.hparams.image_size)
        tang_files = [self.hparams.val_root_tang+x for x in os.listdir(self.hparams.val_root_tang)]
        oulu_files = [self.hparams.val_root_oulu+x for x in os.listdir(self.hparams.val_root_oulu)]
        tang_dataset = load_dataset(
            tang_files, transforms, anti_word='spoof'#, anti_word='fake'
        )
        oulu_dataset = load_dataset(
            oulu_files, transforms, anti_word='spoof'#, anti_word='fake'
        )
        dataset = tang_dataset.concatenate(oulu_dataset)

        dataset = dataset.shuffle(buffer_size = 1024)
        dataset = dataset.batch(batch_size = self.hparams.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#.cache()

        return dataset
