import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd() + '/' + __file__)))
from tsmnet import Stretcher
import torchaudio
from pathlib import Path
import argparse
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', help='e.g. ~/Datasets/gem.wav', type=Path)
    parser.add_argument('weight_dir', help='e.g. ../scripts/logs-fma/weights', type=Path)
    parser.add_argument('-o', '--out_dir', default='samples', help='output directory', type=Path)
    parser.add_argument('-s', '--start', default=None, help='start timestamp', type=int)
    parser.add_argument('-e', '--end', default=None, help='end timestamp', type=int)
    
    args = parser.parse_args()
    return args

def load(file_path, start=None, end=None, verbose=False):
    '''
    args:
      file_path(str): file to load
      start(scalar, optional): start time in second. Default: `None`
      end(scalar, optional): end time in second. Default: `None`
      verbose(bool, optional): print additional messages. Default: `False`
    notes:
      - If you don't understand some syntax, please refer to short-circuiting behavior in operator and, or
    '''
    x, sr = torchaudio.load(file_path)
    x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(x)
    sr = 22050
    
    t_start = start or 0
    t_end   = end   or None
        
    x = x[:, t_start*sr:(end and end*sr)]
    return x, sr

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    stretcher = Stretcher(args.weight_dir)
    x, sr = load(args.audio_path, args.start, args.end)
    
    for rate in [.5, .75, 1, 1.5, 1.75, 2]:
        x_scaled = stretcher(x, rate)
        fname = str(args.out_dir / (str(args.audio_path.stem) + '-' + str(rate) + 'x.wav'))
        print(f'writing {fname}')
        sf.write(fname, x_scaled.T, sr)
         
if __name__ == "__main__":
    main()