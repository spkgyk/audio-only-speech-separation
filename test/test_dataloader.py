###
# Author: Kai Li
# Date: 2022-05-31 16:18:20
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-07-29 06:22:11
###
import yaml
import look2hear.datas
import look2hear.models

from rich import print
from rich.progress import track
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict


def test_LRS2Audio(config):
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(**config["datamodule"]["data_config"])
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader
    for batch in track(train_loader):
        continue
    for batch in track(val_loader):
        continue
    for batch in track(test_loader):
        continue
    for batch in track(train_loader):
        print(batch[0].shape, batch[1].shape, batch[2])
        break
    for batch in track(val_loader):
        print(batch[0].shape, batch[1].shape, batch[2])
        break
    for batch in track(test_loader):
        print(batch[0].shape, batch[1].shape, batch[2])
        break


with open("configs/convtasnet_lrs3.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf)

config, plain_args = parse_args_as_dict(parser, return_plain_args=True)
test_LRS2Audio(config)
