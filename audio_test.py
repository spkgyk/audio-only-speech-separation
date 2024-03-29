###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2022-07-27 22:43:08
###

import os
import yaml
import torch
import argparse
import look2hear.models
import look2hear.datas

from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

import warnings

warnings.filterwarnings("ignore")


def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn,
    )
    # import pdb; pdb.set_trace()
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device
    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx], device=model_device)
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)
            metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf-dir",
        default=None,
        type=str,
        help="Path to model config dump file",
    )

    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    main(arg_dic)
