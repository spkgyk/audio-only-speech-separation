###
# Author: Kai Li
# Date: 2021-06-20 01:32:22
# LastEditors: Please set LastEditors
# LastEditTime: 2022-08-31 21:00:19
###
import yaml
import torch
import argparse
import look2hear.models

from rich import print
from ptflops import get_model_complexity_info
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

with open("configs/unet_base.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)

arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
audiomodel = getattr(look2hear.models, arg_dic["audionet"]["audionet_name"])(
    sample_rate=arg_dic["datamodule"]["data_config"]["sample_rate"], **arg_dic["audionet"]["audionet_config"]
)
# print(v1 == v2)

# start = time.perf_counter()
# for i in range(100):
#     audiomodel(a)
# end = time.perf_counter()
# print((end - start)/100.)
with torch.cuda.device(6):
    a = torch.randn(1, 16000).cuda()
    total_macs = 0
    total_params = 0
    # DPRNN
    model = audiomodel.cuda()
    macs, params = get_model_complexity_info(model, (1, 16000), as_strings=False, print_per_layer_stat=True, verbose=False)
    total_macs += macs
    total_params += params
    # model = nn.Conv1d(1, 65, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (1, 32000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs
    # total_params += params

    # model = nn.ConvTranspose1d(65, 1, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (65, 32000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs*2
    # total_params += params*2

    print(total_macs / 10.0**9)
    print(total_params / 10.0**6)
    # for i in range(1000):
    #     model(a)
