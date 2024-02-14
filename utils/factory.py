from methods.HybridCentric.CLCR import CLCR
from methods.ModelCentric.DER import DER
from methods.DataCentric.FineTune import FineTune
from methods.ModelCentric.FOSTER import FOSTER
from methods.AlgorithmCentric.iCaRL import iCaRL
from methods.AlgorithmCentric.LwF import LwF
from methods.DataCentric.Replay import Replay
from methods.Scratch import Scratch

def get_model(model_name, args):
    name = model_name.lower()
    if name == "iCaRL":
        return iCaRL(args)
    elif name == "Scratch":
        return Scratch(args)
    elif name == "LwF":
        return LwF(args)
    elif name == "FineTune":
        return FineTune(args)
    elif name == "DER":
        return DER(args)
    elif name == "Replay":
        return Replay(args)
    elif name == "CLCR":
        return CLCR(args)
    elif name == "FOSTER":
        return FOSTER(args)
    else:
        assert 0
