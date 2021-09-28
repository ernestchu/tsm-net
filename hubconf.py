dependencies = ["torch", "yaml"]
from tsmnet import Neuralgram


def load_tsmnet(model_name="general"):
    """
    Exposes a TSM-Net Interface
    Args:
        model_name (str): Supports only 1 models, 'general'
    Returns:
        object (Neuralgram):  Neuralgram class.
            Default function (___call__) converts raw audio to neuralgram
            inverse function convert neuralgram to raw audio
    """

    return Neuralgram(path=None, github=True, model_name=model_name)
