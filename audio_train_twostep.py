###
# Author: Kai Li
# Date: 2022-04-06 14:51:43
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-06-13 12:10:15
###
import os
import json
import torch
import argparse
import look2hear.datas
import look2hear.utils
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics

import pytorch_lightning as pl

from look2hear.system import make_optimizer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import *
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Full path to save best validation model",
)


def update_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "sm" in k:
            update_dict[k] = v
            # import pdb; pdb.set_trace()
        else:
            pass
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model


def main(config):
    print_only("Instantiating datamodule <{}>".format(config["datamodule"]["data_name"]))
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(**config["datamodule"]["data_config"])
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    # Define model and optimizer
    print_only("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    state_dict = torch.load(config["training"]["pretrain_dir"], map_location="cpu")["state_dict"]
    model = update_parameter(model, state_dict)
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only("Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"]))
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"]
        )

    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"])
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only("Instantiating Loss, Train <{}>, Val <{}>".format(config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]))
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        accelerator=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    main(arg_dic)
