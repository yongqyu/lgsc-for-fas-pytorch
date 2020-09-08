import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''
from argparse import ArgumentParser, Namespace
import safitty
from typing import List, Tuple, Union
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from pl_model import LightningModel
from datasets import get_test_augmentations, load_dataset
from metrics import eval_from_scores
from utils import set_gmem_growth

set_gmem_growth()


def prepare_infer_dataloader(args: Namespace) -> tf.data.Dataset:
    transforms = get_test_augmentations
    # total_files = [args.root+sub_path+'/'+x for sub_path in os.listdir(args.root)
    #                                         for x in os.listdir(args.root+sub_path)]
    total_files = [args.root+x for x in os.listdir(args.root)]

    dataset = load_dataset(
        total_files, transforms, real_word='real'
    )

    dataset = dataset.batch(batch_size = args.batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#.cache()

    return dataset


def load_model_from_checkpoint(args_: Namespace) -> LightningModel:
    model = LightningModel(args_)
    model.load_weights(args_.checkpoints)
    # model.eval()
    # model.to(device)
    return model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    return Namespace(**configs)


def infer_model(
    model: LightningModel,
    dataloader: tf.data.Dataset,
    verbose: bool = False,
    with_labels: bool = True,
) -> Union[Tuple[float, float, float, float, float], List[float]]:
    scores = []
    targets= []

    if verbose:
        dataloader = tqdm(dataloader)
    f = open('kp_test_texture_result.txt', 'w')
    for batch in dataloader:
        if with_labels:
            images, labels, names = batch
            labels = tf.cast(labels, tf.float32)
        cues = model.infer(images)

        for i in range(cues.shape[0]):
            score = tf.reduce_mean(cues[i, ...])
            scores.append(score)
            f.write(f'{score} {int(labels[i])} {names[i]}\n')
        if with_labels:
            targets.append(labels)
    f.close()

    if with_labels:
        metrics_, best_thr, acc = eval_from_scores(
            np.array(scores), tf.concat(targets, axis=0).numpy()
        )
        acer, apcer, npcer = metrics_
        if verbose:
            print(f"ACER: {acer}")
            print(f"APCER: {apcer}")
            print(f"NPCER: {npcer}")
            print(f"Best accuracy: {acc}")
            print(f"At threshold: {best_thr}")
        return acer, apcer, npcer, acc, best_thr
    else:
        return scores


if __name__ == "__main__":
    args_ = parse_args()
    model_ = load_model_from_checkpoint(args_)

    dataloader_ = prepare_infer_dataloader(args_)

    if args_.with_labels:
        acer_, apcer_, npcer_, acc_, best_thr_ = infer_model(
            model_, dataloader_, args_.verbose, True
        )
        with open(args_.out_file, "w") as file:
            file.write(f"acer - {acer_}\n")
            file.write(f"apcer - {apcer_}\n")
            file.write(f"npcer - {npcer_}\n")
            file.write(f"acc - {acc_}\n")
            file.write(f"best_thr - {best_thr_}\n")

    else:
        scores_ = infer_model(model_, dataloader_, False, False)
        # if you don't have answers you can write your scores into some file
        with open(args_.out_file, "w") as file:
            file.write("\n".join(list(map(lambda x: str(x), scores_))))
