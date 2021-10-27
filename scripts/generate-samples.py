import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd() + '/' + __file__)))
from tsmnet import Stretcher
import torch, torchaudio
from pathlib import Path
import numpy as np
import librosa
import sox
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_list', help='A file containing audio filenames. One song each line.', type=Path)
    parser.add_argument('weight_dir', help='e.g. ../scripts/logs-fma/weights', type=Path)
    parser.add_argument('CR', help='Compression ratio. e.g. 1024. Can also be `librosa` or `sox`', type=str)
    parser.add_argument('-o', '--out_dir', default='samples', help='output directory', type=Path)
    
    args = parser.parse_args()
    return args

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [Path(f.rstrip()) for f in files]
    return files

def main():
    args = parse_args()
    # initialize different stretchers
    if str(args.weight_dir) == 'sox':
        tfm = sox.transform.Transformer()
    elif str(args.weight_dir) != 'librosa':
        stretcher = Stretcher(args.weight_dir)
        
    files = files_to_list(args.audio_list)

    for file in files:
        (args.out_dir / file.stem / args.CR).mkdir(parents=True, exist_ok=True)
        x, sr = torchaudio.load(file)
        x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(x)
        sr = 22050
        if str(args.weight_dir) == 'librosa' or str(args.weight_dir) == 'sox':
            x = x.numpy()

        for rate in ['0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']:
            if str(args.weight_dir) == 'librosa':
                if len(x.shape) == 1:
                    x_scaled = librosa.effects.time_stretch(x, float(rate))
                else:
                    x_scaled = [ librosa.effects.time_stretch(x[0], float(rate)) ]
                    for i in range(1, len(x.shape)):
                        x_scaled.append(librosa.effects.time_stretch(x[i], float(rate)))
                    x_scaled = np.array(x_scaled)
            elif str(args.weight_dir) == 'sox':
                tfm.tempo(float(rate))
                if len(x.shape) == 1:
                    x_scaled = tfm.build_array(input_array=x, sample_rate_in=sr)
                else:
                    x_scaled = [ tfm.build_array(input_array=x[0], sample_rate_in=sr) ]
                    for i in range(1, len(x.shape)):
                        x_scaled.append(tfm.build_array(input_array=x[i], sample_rate_in=sr))
                    x_scaled = np.array(x_scaled)
            else:
                x_scaled = stretcher(x, float(rate))
            fname = str(args.out_dir / file.stem / args.CR / (rate + '.mp3'))
            print(f'writing {fname}')
            torchaudio.save(fname, torch.from_numpy(x_scaled), sr)
            
if __name__ == "__main__":
    main()