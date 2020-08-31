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
        self.log_cues = not self.hparams.cue_log_every == 0
        self.grid_maker = GridMaker()
        if self.hparams.use_focal_loss:
            self.clf_criterion = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            self.clf_criterion = K.losses.SparseCategoricalCrossentropy()

    def forward(self, x):
        return self.model(x)

    def infer(self, x):
        outs, _ = self.model(x)
        return outs[-1]

    def calc_losses(self, outs, clf_out, target):

        clf_loss = (
            self.clf_criterion(clf_out, target)
            * self.hparams.loss_coef["clf_loss"]
        )
        cue = outs[-1]
        cue = target.reshape(-1, 1, 1, 1) * cue
        num_reg = (
            tf.math.reduce_sum(target) * cue.shape[1] * cue.shape[2] * cue.shape[3]
        ).type(tf.float32)
        reg_loss = (
            tf.math.reduce_sum(tf.math.abs(cue)) / (num_reg + 1e-9)
        ) * self.hparams.loss_coef["reg_loss"]

        trip_loss = 0
        bs = outs[-1].shape[0]
        for feat in outs[:-1]:
            feat = tf.nn.avg_pool2d(feat, [1, 1]).reshape(bs, -1)
            trip_loss += (
                self.triplet_loss(feat, target)
                * self.hparams.loss_coef["trip_loss"]
            )
        total_loss = clf_loss + reg_loss + trip_loss

        return total_loss

    def training_step(self, batch):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = tf.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "train_avg_loss": avg_loss,
        }
        return {"train_avg_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)
        val_dict = {
            "val_loss": loss,
            "score": clf_out.cpu().numpy(),
            "target": target.cpu().numpy(),
        }
        if self.log_cues:
            if (
                self.current_epoch * batch_idx
            ) % self.hparams.cue_log_every == 0:
                cues_grid, images_grid = self.grid_maker(
                    input_.detach().cpu()[:6], outs[-1][:6]
                )
                self.logger.experiment.add_image(
                    "cues", cues_grid, self.current_epoch * batch_idx
                )
                self.logger.experiment.add_image(
                    "images", images_grid, self.current_epoch * batch_idx
                )

        return val_dict

    def validation_epoch_end(self, outputs):
        avg_loss = tf.stack([x["val_loss"] for x in outputs]).mean()
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
            initial_learning_rate=self.hparams.lr, decay_steps=self.hparams.milestones, decay_rate=self.hparams.gamma
        )
        optim = K.optimizers.Adam(scheduler)
        return optim

    def train_dataloader(self):
        transforms = get_train_augmentations(self.hparams.image_size)
        # total_frames = [self.hparams.train_root+sub_dir+'/'+x
        #                                    for sub_dir in os.listdir(self.hparams.train_root)
        #                                    for x in os.listdir(self.hparams.train_root+sub_dir)]
        total_frames = [self.hparams.train_root+x
                                           for x in os.listdir(self.hparams.train_root)]
        dataset = load_dataset(
            total_frames, self.hparams.train_root, transforms, batch_size=self.hparams.batch_size, anti_word='anti'
        )
        # if self.hparams.use_balance_sampler:
        #     labels = list(df.target.values)
        #     sampler = BalanceClassSampler(labels, mode="upsampling")
        #     shuffle = False
        # else:
        sampler = None
        shuffle = True

        # if shuffle: dataset = dataset.shuffle(dataset.shape())
        return dataset

    def val_dataloader(self):
        transforms = get_test_augmentations(self.hparams.image_size)
        total_frames = [self.hparams.val_root+x for x in os.listdir(self.hparams.val_root)]
        dataset = load_dataset(
            total_frames, self.hparams.val_root, transforms, batch_size=self.hparams.batch_size, anti_word='spoof'
        )
        return dataset
