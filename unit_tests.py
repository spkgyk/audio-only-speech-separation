import torch

from tqdm import trange
from time import time
from look2hear.models import TDANet, AFRCNN, TasNet, Sepformer, ConvTasNet, DPRNNTasNet, Sandglasset
from look2hear.losses import PITLossWrapper, pairwise_neg_snr
from look2hear.system import make_optimizer
import warnings

warnings.filterwarnings("ignore")


def test_model(model, batch_size=2, length=16000, dry=False, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    epochs = 5
    loss = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    optimizer = make_optimizer(model.parameters(), optim_name="adam", lr=0.001, weight_decay=0)
    model = model.to(device)

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
            pbar.set_description("Model: {}, time taken: {:.2f}".format(model.model_name, time_s))


if __name__ == "__main__":

    test_model(TDANet(), batch_size=4, length=32000, dry=True)
    test_model(TDANet(), batch_size=4, length=32000, device="cpu", dry=True)

    print("\nGPU Results")

    test_model(TDANet())
    test_model(AFRCNN())
    test_model(Sepformer())
    test_model(ConvTasNet())
    test_model(Sandglasset())
    test_model(DPRNNTasNet())
    test_model(TasNet(module="TCN"))
    test_model(TasNet(module="DPRNN"))
    test_model(TasNet(module="DPTNet"))
    test_model(TasNet(module="SudoRMRF"))
    test_model(TasNet(module="GC_TCN"))
    test_model(TasNet(module="GC_DPRNN"))
    test_model(TasNet(module="GC_DPTNet"))
    test_model(TasNet(module="GC_SudoRMRF"))

    print("\n\nCPU Results")

    test_model(TDANet(), device="cpu")
    test_model(AFRCNN(), device="cpu")
    test_model(Sepformer(), device="cpu")
    test_model(ConvTasNet(), device="cpu")
    test_model(Sandglasset(), device="cpu")
    test_model(DPRNNTasNet(), device="cpu")
    test_model(TasNet(module="TCN"), device="cpu")
    test_model(TasNet(module="DPRNN"), device="cpu")
    test_model(TasNet(module="DPTNet"), device="cpu")
    test_model(TasNet(module="SudoRMRF"), device="cpu")
    test_model(TasNet(module="GC_TCN"), device="cpu")
    test_model(TasNet(module="GC_DPRNN"), device="cpu")
    test_model(TasNet(module="GC_DPTNet"), device="cpu")
    test_model(TasNet(module="GC_SudoRMRF"), device="cpu")
