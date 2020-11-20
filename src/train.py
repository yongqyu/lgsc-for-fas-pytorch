import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''
from time import gmtime, strftime
import numpy as np
from argparse import ArgumentParser, Namespace
import safitty
import tensorflow as tf

from pl_model import LightningModel, strategy
from utils import set_gmem_growth

set_gmem_growth()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    with strategy.scope():
        model = LightningModel(hparams=configs)
        #model.load_weights('./oulu_logs/epoch_1.ckpt')
        optim = model.configure_optimizers()

        logdir = "/data/private/ASD/tb/"
        writer = tf.summary.create_file_writer(logdir)
        writer.set_as_default()

        train_dataloader = model.train_dataloader()
        val_dataloader = strategy.experimental_distribute_dataset(model.val_dataloader())
        val_min_acer = 1.
        for epoch in range(1, configs.max_epochs):
            print(f'[Epoch {epoch}]')
            if epoch == 5:
                model.model.backbone.unfreeze_encoder()
                model.model.clf.unfreeze_clf()

            tr_outputs = []
            for batch_idx, batch in enumerate(train_dataloader, 1):
                tr_outputs.append(strategy.run(model.training_step, args=(batch,)))

                if batch_idx % configs.cue_log_every == 0:
                    val_loss, val_acc, val_acer = [], [], []
                    for val_batch_idx, (val_batch) in enumerate(val_dataloader, 1):
                        v_loss, v_acc, v_acer = strategy.run(model.validation_step, args=(val_batch,))
                        val_acc.append(np.mean(v_acc.values))
                        val_acer.append(np.mean(v_acer.values))

                    # [tf.summary.scalar(k, v, step=epoch) for k,v in val_tb_log['log'].items()]
                    tr_out_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, tr_outputs[-1]['loss'], axis=0)
                    print(f"[Batch {batch_idx}] Train loss : {tr_out_loss}\tVal acc : {np.mean(val_acc)}\tVal acer : {np.mean(val_acer)}\t{strftime('%Y-%m-%d %H:%M:%S', gmtime())}")

                    if np.mean(val_acer) < val_min_acer:
                        val_min_acer = np.mean(val_acer)
                        model.save_weights(f'oulu_logs/epoch_{epoch}.ckpt')

            # tr_output = strategy.run(model.training_epoch_end, args=(tr_outputs,))
            # [tf.summary.scalar(k, v, step=epoch) for k,v in tr_output['log'].items()]
