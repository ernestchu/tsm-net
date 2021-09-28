import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from tsmnet.dataset import AudioDataset
from tsmnet.modules import Autoencoder, Discriminator
from tsmnet.utils import save_sample

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

import numpy as np
import time
import argparse
from pathlib import Path
import yaml

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--compress_ratios", default='22488')
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=1)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with wandb.init(project=args.project, entity='tsm-net', config=args):
        root = Path(args.save_path)
        load_root = Path(args.load_path) if args.load_path else None
        (root / 'original').mkdir(parents=True, exist_ok=True)
        (root / 'generated').mkdir(parents=True, exist_ok=True)
        (root / 'weights').mkdir(parents=True, exist_ok=True)
        (root / 'ONNX').mkdir(parents=True, exist_ok=True)
        
        with open(root / "weights/args.yml", "w") as f:
            yaml.dump(args, f)
        
        #######################
        # Load PyTorch Models #
        #######################
        netA = Autoencoder([int(n) for n in args.compress_ratios] ,args.ngf, args.n_residual_layers).cuda()
        netD = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor).cuda()

        wandb.watch((netA, netD), log='all', log_freq=args.log_interval)

        #####################
        # Create optimizers #
        #####################
        optA = torch.optim.Adam(netA.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if load_root and load_root.exists():
            netA.load_state_dict(torch.load(load_root / "netA.pt"))
            optA.load_state_dict(torch.load(load_root / "optA.pt"))
            netD.load_state_dict(torch.load(load_root / "netD.pt"))
            optD.load_state_dict(torch.load(load_root / "optD.pt"))

        #######################
        # Create data loaders #
        #######################
        train_set = AudioDataset(
            Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=22050
        )
        test_set = AudioDataset(
            Path(args.data_path) / "test_files.txt",
            22050 * 4,
            sampling_rate=22050,
            augment=False,
        )

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=1)

        ##########################
        # Dumping original audio #
        ##########################
        test_audio = []
        logs = {}
        for i, x_t in enumerate(test_loader):
            x_t = x_t.cuda()

            test_audio.append(x_t)

            audio = x_t.squeeze().cpu()
#             torchaudio.save(str(root / f'original/sample-{i}.wav'), audio, 22050)
            save_sample(root / f'original/sample-{i}.wav', 22050, audio)
            logs[f'sample-{i}'] = wandb.Audio(audio, 22050, f'sample-{i}')

            if i == args.n_test_samples - 1:
                break
        wandb.log({'original': logs})

        costs = []
        start = time.time()

        # enable cudnn autotuner to speed up training
        torch.backends.cudnn.benchmark = True

        best_audio_reconst = 1000000
        steps = 0
        for epoch in range(1, args.epochs + 1):
            for iterno, x_t in enumerate(train_loader):
                x_t = x_t.cuda()
                x_pred_t = netA(x_t)

                with torch.no_grad():
                    x_error = F.l1_loss(x_t, x_pred_t.detach()).item()
                    s_t = netA.encoder(x_t)
                    s_pred_t = netA.encoder(x_pred_t.detach())
                    s_error = F.l1_loss(s_t, s_pred_t).item()

                #######################
                # Train Discriminator #
                #######################
                D_fake_det = netD(x_pred_t.detach())
                D_real = netD(x_t)

                loss_D = 0
                for scale in D_fake_det:
                    loss_D += F.relu(1 + scale[-1]).mean()

                for scale in D_real:
                    loss_D += F.relu(1 - scale[-1]).mean()

                netD.zero_grad()
                loss_D.backward()
                optD.step()

                #####################
                # Train Autoencoder #
                #####################
                D_fake = netD(x_pred_t)

                loss_A = 0
                for scale in D_fake:
                    loss_A += -scale[-1].mean()

                loss_feat = 0
                feat_weights = 4.0 / (args.n_layers_D + 1)
                D_weights = 1.0 / args.num_D
                wt = D_weights * feat_weights
                for i in range(args.num_D):
                    for j in range(len(D_fake[i]) - 1):
                        loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

                netA.zero_grad()
                (loss_A + args.lambda_feat * loss_feat).backward()
                optA.step()

                ######################
                # Update wandb #
                ######################
                costs.append([loss_D.item(), loss_A.item(), loss_feat.item(), x_error, s_error])

                wandb.log({'loss': {
                    'discriminator': costs[-1][0],
                    'autoencoder': costs[-1][1],
                    'feature-matching': costs[-1][2],
                    'audio-reconstruction': costs[-1][3],
                    'mel-reconstruction': costs[-1][4]
                }})
                steps += 1

                if steps % args.save_interval == 0:
                    st = time.time()
                    with torch.no_grad():
                        logs = {}
                        for i, audio in enumerate(test_audio):
                            pred_audio = netA(audio)
                            pred_audio = pred_audio.squeeze().cpu()
#                             torchaudio.save(str(root / f'generated/sample-{i}.wav'), pred_audio, 22050)
                            save_sample(root / f'generated/sample-{i}.wav', 22050, pred_audio)
                            logs[f'sample-{i}'] = wandb.Audio(pred_audio, 22050, f'sample-{i}')
                        wandb.log({'generated': logs})

                    torch.save(netA.state_dict(), root / "weights/netA.pt")
                    torch.save(optA.state_dict(), root / "weights/optA.pt")

                    torch.save(netD.state_dict(), root / "weights/netD.pt")
                    torch.save(optD.state_dict(), root / "weights/optD.pt")

                    if np.asarray(costs).mean(0)[-2] < best_audio_reconst:
                        best_audio_reconst = np.asarray(costs).mean(0)[-2]
                        print("saved the best model")
                        torch.save(netA.state_dict(), root / "weights/best_netA.pt")
                        torch.save(netD.state_dict(), root / "weights/best_netD.pt")
                        
                        with torch.no_grad():
                            # Save the model in the exchangeable ONNX format
                            torch.onnx.export(netA, test_audio[0], root / 'ONNX/best_netA.onnx')
                            torch.onnx.export(netD, netA(test_audio[0]), root / 'ONNX/best_netD.onnx')
                            wandb.save(str(root / "ONNX/best_netA.onnx"))
                            wandb.save(str(root / "ONNX/best_netD.onnx"))

                    print("Took %5.4fs to generate samples" % (time.time() - st))
                    print("-" * 100)

                if steps % args.log_interval == 0:
                    print(
                        "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                            epoch,
                            iterno,
                            len(train_loader),
                            1000 * (time.time() - start) / args.log_interval,
                            np.asarray(costs).mean(0),
                        )
                    )
                    costs = []
                    start = time.time()


if __name__ == "__main__":
    main()
