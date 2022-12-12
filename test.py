import torch
from look2hear.models import TasNet


def test_model(model):
    x = torch.rand(2, 64000)  # (batch, length)
    y = model(x)
    print(y.shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model_GC_TCN = TasNet(module="GC_TCN")
    model_GC_DPRNN = TasNet(module="GC_DPRNN")
    model_GC_DPTNet = TasNet(module="GC_DPTNet")
    model_GC_SudoRMRF = TasNet(module="GC_SudoRMRF")
    model_TCN = TasNet(module="TCN")
    model_DPRNN = TasNet(module="DPRNN")
    model_DPTNet = TasNet(module="DPTNet")
    model_SudoRMRF = TasNet(module="SudoRMRF")

    test_model(model_GC_SudoRMRF)
    test_model(model_GC_DPTNet)
    test_model(model_GC_DPRNN)
    test_model(model_GC_TCN)
    test_model(model_SudoRMRF)
    test_model(model_DPTNet)
    test_model(model_DPRNN)
    test_model(model_TCN)
