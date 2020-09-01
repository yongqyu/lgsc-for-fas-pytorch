import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''
from argparse import ArgumentParser, Namespace
import safitty

from pl_model import LightningModel


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", required=True)
    args = parser.parse_args()
    configs = safitty.load(args.configs)
    configs = Namespace(**configs)

    model = LightningModel(hparams=configs)
    optim = model.configure_optimizers()

    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()
    tr_min_loss = 1.
    for epoch in range(1, configs.max_epochs):
        tr_outputs = []
        for batch_idx, batch in enumerate(train_dataloader, 1):
            tr_outputs.append(model.training_step(batch))

            if batch_idx % configs.cue_log_every == 0:
                val_outputs = []
                for val_batch_idx, val_batch in enumerate(1, val_dataloader):
                    val_outputs.append(model.validation_step(val_batch))
                val_tb_log = model.validation_epoch_end(val_outputs)
                print(val_tb_log['log'])

        tr_output = model.training_epoch_end(tr_outputs)
        if tf_output['train_avg_loss'] < tr_min_loss:
            tr_min_loss = tf_output['train_avg_loss']
            model.save(f'./lightning_logs/epoch_{epoch}')
