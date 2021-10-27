#!/bin/bash
python3 generate-samples.py song-lists/fma.txt logs/logs-fma/weights/ 1024 -o "samples/fma"
python3 generate-samples.py song-lists/fma.txt logs/logs-fma-512x/weights/ 512 -o "samples/fma"
python3 generate-samples.py song-lists/fma.txt logs/logs-fma-256x/weights/ 256 -o "samples/fma"
python3 generate-samples.py song-lists/fma.txt librosa librosa -o "samples/fma"
python3 generate-samples.py song-lists/fma.txt sox sox -o "samples/fma"

python3 generate-samples.py song-lists/musicnet.txt logs/logs-musicnet/weights/ 1024 -o "samples/musicnet"
python3 generate-samples.py song-lists/musicnet.txt logs/logs-musicnet-512x/weights/ 512 -o "samples/musicnet"
python3 generate-samples.py song-lists/musicnet.txt logs/logs-musicnet-256x/weights/ 256 -o "samples/musicnet"
python3 generate-samples.py song-lists/musicnet.txt librosa librosa -o "samples/musicnet"
python3 generate-samples.py song-lists/musicnet.txt sox sox -o "samples/musicnet"

python3 generate-samples.py song-lists/vctk.txt logs/logs-vctk/weights/ 1024 -o "samples/vctk"
python3 generate-samples.py song-lists/vctk.txt logs/logs-vctk-512x/weights/ 512 -o "samples/vctk"
python3 generate-samples.py song-lists/vctk.txt logs/logs-vctk-256x/weights/ 256 -o "samples/vctk"
python3 generate-samples.py song-lists/vctk.txt librosa librosa -o "samples/vctk"
python3 generate-samples.py song-lists/vctk.txt sox sox -o "samples/vctk"

python3 generate-samples.py song-lists/yang.txt logs/logs-yang/weights/ 1024 -o "samples/yang"
python3 generate-samples.py song-lists/yang.txt logs/logs-yang-512x/weights/ 512 -o "samples/yang"
python3 generate-samples.py song-lists/yang.txt logs/logs-yang-256x/weights/ 256 -o "samples/yang"
python3 generate-samples.py song-lists/yang.txt librosa librosa -o "samples/yang"
python3 generate-samples.py song-lists/yang.txt sox sox -o "samples/yang"