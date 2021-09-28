from tsmnet.modules import Autoencoder

from pathlib import Path
import yaml
import torch
import os


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    root = Path(path)
    with open(root / "args.yml", "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    netA = Autoencoder([int(n) for n in args.compress_ratios], args.ngf, args.n_residual_layers).to(device)
    netA.load_state_dict(torch.load(root / "best_netA.pt", map_location=device))
    return netA


class Neuralgram:
    def __init__(
        self,
        path,
        device=get_default_device(),
        github=False,
        model_name="general",
    ):
        if github:
            self.netA = Encoder(32, 1).to(device)
            root = Path(os.path.dirname(__file__)).parent
            self.netA.load_state_dict(
                torch.load(root / f"models/{model_name}.pt", map_location=device)
            )
        else:
            self.netA = load_model(path, device)
        self.device = device

    def __call__(self, audio):
        """
        Performs audio to neuralgram conversion (See Autoencoder.encoder in tsmnet/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: neuralgram computed on input audio (batch_size, channels, timesteps)
        """
        with torch.no_grad():
            return self.netA.encoder(audio.unsqueeze(1).to(self.device))

    def inverse(self, neu):
        """
        Performs neuralgram to audio conversion
        Args:
            neu (torch.tensor): PyTorch tensor containing neuralgram (batch_size, channels, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)

        """
        with torch.no_grad():
            return self.netA.decoder(neu.to(self.device)).squeeze(1)
