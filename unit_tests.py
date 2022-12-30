import torch

from tqdm import trange
from time import time
from look2hear.models import TDANet, AFRCNN, TasNet, Sepformer, ConvTasNet, DPRNNTasNet, Sandglasset, BSRNN
from look2hear.losses import PITLossWrapper, pairwise_neg_snr
from look2hear.system import make_optimizer
from ptflops import get_model_complexity_info
import warnings

warnings.filterwarnings("ignore")


def test_model(model, batch_size=4, length=32000, dry=False, device=None, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    loss = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    optimizer = make_optimizer(model.parameters(), optim_name="adam", lr=0.001, weight_decay=0)
    model = model.to(device)

    total_macs = 0
    total_params = 0
    macs, params = get_model_complexity_info(model, (1, length), as_strings=False, print_per_layer_stat=False, verbose=False)
    model.train()
    total_macs += macs / 10.0**6
    total_params += params / 10.0**6

    s = time()
    pbar = trange(epochs) if not dry else range(epochs)
    for _ in pbar:
        x = torch.rand(batch_size, length).to(device)
        y = torch.rand(batch_size, 2, length).to(device)
        y_hat = model(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        time_s = time() - s
        if not dry:
            pbar.set_description(
                "Model - {:<15} Time Taken (s) - {:<8} Device - {:<8} MACs (M) - {:<8} Params (M) - {:<8}".format(
                    model.model_name, str(round(time_s, 2)), str(device), str(round(total_macs, 2)), str(round(total_params, 2))
                )
            )


if __name__ == "__main__":

    test_model(TDANet(), dry=True)
    # test_model(TDANet(), batch_size=4, length=32000, device="cpu", dry=True)

    print("\nGPU Results")

    test_model(BSRNN(sample_rate=8000, win=256, stride=64), epochs=50)
    test_model(BSRNN(sample_rate=16000, win=1024, stride=256), epochs=50)
    test_model(BSRNN(sample_rate=44100, win=1024, stride=256), epochs=50)
    print()

    test_model(TasNet(module="DPTNet", enc_dim=256, bn_dim=64), epochs=50)
    test_model(TasNet(module="DPTNet", enc_dim=64, bn_dim=64), epochs=50)
    # print()

    # test_model(TasNet(module="Unfolded_DPTNet", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="Unfolded_DPTNet", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="GC_DPTNet", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="GC_DPTNet", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="DPRNN", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="DPRNN", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="Unfolded_DPRNN", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="Unfolded_DPRNN", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="GC_DPRNN", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="GC_DPRNN", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="TCN", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="TCN", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="GC_TCN", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="GC_TCN", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="SudoRMRF", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="SudoRMRF", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TasNet(module="GC_SudoRMRF", enc_dim=256, bn_dim=64), epochs=100)
    # test_model(TasNet(module="GC_SudoRMRF", enc_dim=64, bn_dim=64), epochs=100)
    # print()

    # test_model(TDANet())
    # test_model(AFRCNN())
    # test_model(Sepformer())
    # test_model(ConvTasNet())
    # test_model(Sandglasset())
    # test_model(DPRNNTasNet())

    # print("\n\nCPU Results")

    # test_model(TDANet(), device="cpu")
    # test_model(AFRCNN(), device="cpu")
    # test_model(Sepformer(), device="cpu")
    # test_model(ConvTasNet(), device="cpu")
    # test_model(Sandglasset(), device="cpu")
    # test_model(DPRNNTasNet(), device="cpu")
    # test_model(TasNet(module="TCN"), device="cpu")
    # test_model(TasNet(module="DPRNN"), device="cpu")
    # test_model(TasNet(module="DPTNet"), device="cpu")
    # test_model(TasNet(module="SudoRMRF"), device="cpu")
    # test_model(TasNet(module="GC_TCN"), device="cpu")
    # test_model(TasNet(module="GC_DPRNN"), device="cpu")
    # test_model(TasNet(module="GC_DPTNet"), device="cpu")
    # test_model(TasNet(module="GC_SudoRMRF"), device="cpu")
