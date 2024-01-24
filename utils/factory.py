from methods.CLCR import CLCR
from methods.DER import DER
from methods.FineTune import FineTune
from methods.FOSTER import FOSTER
from methods.iCaRL import iCaRL
from methods.LwF import LwF
from methods.Replay import Replay
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
