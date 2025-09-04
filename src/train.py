# src/train.py
import argparse, itertools, os, random, glob, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

def conv(in_c, out_c, ks=4, s=2, pad=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, ks, s, pad, bias=not norm)]
    if norm: layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    # 3×256×256  ->  3×256×256
    def __init__(self, n_res=6):
        super().__init__()
        model = [nn.Conv2d(3, 64, 7, 1, 3, bias=False),
                 nn.InstanceNorm2d(64), nn.ReLU(True)]
        # down
        c = 64
        for _ in range(2):
            model += [nn.Conv2d(c, c*2, 3, 2, 1, bias=False),
                      nn.InstanceNorm2d(c*2), nn.ReLU(True)]
            c *= 2
        # residual
        for _ in range(n_res): model += [ResBlock(c)]
        # up
        for _ in range(2):
            model += [nn.ConvTranspose2d(c, c//2, 3, 2, 1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(c//2), nn.ReLU(True)]
            c //= 2
        model += [nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    # PatchGAN 70×70
    def __init__(self):
        super().__init__()
        layers = []
        cs = [3, 64, 128, 256, 512]
        layers += conv(cs[0], cs[1], norm=False)
        layers += conv(cs[1], cs[2])
        layers += conv(cs[2], cs[3])
        layers += [nn.Conv2d(cs[3], cs[4], 4, 1, 1, bias=False),
                   nn.InstanceNorm2d(cs[4]),
                   nn.LeakyReLU(0.2, True),
                   nn.Conv2d(cs[4], 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)
# -----------------------------------------------------------

class ImageFolder(Dataset):
    def __init__(self, root, domain, size=256):
        # Adapted for SageMaker's data directory structure
        self.paths = sorted(glob.glob(os.path.join(root, domain, "*")))
        self.t = T.Compose([T.Resize((size, size)),
                            T.ToTensor(),
                            T.Normalize((0.5,)*3, (0.5,)*3)])

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img)

@torch.no_grad()
def sample(generator, batchA, step, out_dir):
    fakeB = generator(batchA).detach().cpu()*0.5+0.5
    realA = batchA.detach().cpu()*0.5+0.5
    grid = torch.cat([realA, fakeB], dim=0)
    save_image(grid, os.path.join(out_dir, f"{step:06}.jpg"), nrow=len(realA))

# --- (save_checkpoint, save_final_checkpoint, and load_checkpoint functions remain unchanged) ---
def save_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, out_dir, metrics):
    """Save a training checkpoint"""
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'G_AB': G_AB.state_dict(),
        'G_BA': G_BA.state_dict(),
        'D_A': D_A.state_dict(),
        'D_B': D_B.state_dict(),
        'g_opt': g_opt.state_dict(),
        'd_opt': d_opt.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint_step_{step}.pth")

    # Also save just the generators for easy inference
    torch.save(G_AB.state_dict(), f"{out_dir}/G_A2B_{step}.pth")
    torch.save(G_BA.state_dict(), f"{out_dir}/G_B2A_{step}.pth")

def save_final_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, out_dir, metrics):
    """Save final models and training history"""
    # Save complete checkpoint
    checkpoint = {
        'step': step,
        'epoch': epoch + 1,
        'G_AB': G_AB.state_dict(),
        'G_BA': G_BA.state_dict(),
        'D_A': D_A.state_dict(),
        'D_B': D_B.state_dict(),
        'g_opt': g_opt.state_dict(),
        'd_opt': d_opt.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint_final.pth")

    # Save just generators
    torch.save(G_AB.state_dict(), f"{out_dir}/G_A2B_final.pth")
    torch.save(G_BA.state_dict(), f"{out_dir}/G_B2A_final.pth")

    # Save training history
    with open(f"{out_dir}/training_history.json", 'w') as f:
        json.dump({
            'g_losses': metrics['g_losses'],
            'd_losses': metrics['d_losses'],
            'best_g_loss': metrics['best_g_loss'],
            'total_steps': step,
            'total_epochs': epoch + 1
        }, f, indent=2)

def load_checkpoint(checkpoint_path, G_AB, G_BA, D_A, D_B, g_opt, d_opt):
    """Resume training from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    D_A.load_state_dict(checkpoint['D_A'])
    D_B.load_state_dict(checkpoint['D_B'])
    g_opt.load_state_dict(checkpoint['g_opt'])
    d_opt.load_state_dict(checkpoint['d_opt'])
    print(f"Resumed from epoch {checkpoint['epoch']} at step {checkpoint['step']}")
    return checkpoint['epoch'], checkpoint['step'], checkpoint.get('metrics', {})

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # SageMaker's output directory. Your script's --out argument is now controlled by SageMaker.
    output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # Networks and Optimizers must be defined before loading a checkpoint
    G_AB, G_BA = Generator().to(device), Generator().to(device)
    D_A, D_B = Discriminator().to(device), Discriminator().to(device)
    g_opt = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))

    # --- Resume from Checkpoint Logic ---
    start_epoch, step = 0, 0
    metrics = {'g_losses': [], 'd_losses': [], 'best_g_loss': float('inf')}

    if args.resume:
        # SageMaker automatically downloads checkpoint data to /opt/ml/checkpoints
        resume_file_path = os.path.join(args.checkpoint_dir, args.resume)
        if os.path.exists(resume_file_path):
            start_epoch, step, metrics = load_checkpoint(resume_file_path, G_AB, G_BA, D_A, D_B, g_opt, d_opt)
        else:
            print(f"WARNING: Checkpoint file not found at {resume_file_path}. Starting from scratch.")

    # Datasets and Dataloaders using SageMaker's data input directory
    A = ImageFolder(args.data_dir, domain="trainA")
    B = ImageFolder(args.data_dir, domain="trainB")
    loaderA = DataLoader(A, args.bs, True, drop_last=True, num_workers=args.num_workers)
    loaderB = DataLoader(B, args.bs, True, drop_last=True, num_workers=args.num_workers)

    # Dynamic calculation of intervals
    steps_per_epoch = min(len(A)//args.bs, len(B)//args.bs)
    total_steps = args.epochs * steps_per_epoch
    sample_interval = max(1, steps_per_epoch // args.samples_per_epoch) if args.samples_per_epoch > 0 else 0
    checkpoint_interval = steps_per_epoch * args.checkpoint_epochs if args.checkpoint_epochs > 0 else 0

    print(f"Training on {device.upper()}. Outputting to {output_dir}")

    for epoch in range(start_epoch, args.epochs):
        # ... (Your training loop code is identical here) ...
        # The key change is that all file saving operations now use 'output_dir'
        # which points to SageMaker's model directory.
        pbar = tqdm(zip(loaderA, loaderB), desc=f"Epoch {epoch+1}/{args.epochs}", total=steps_per_epoch)
        for realA, realB in pbar:
             # ... training steps for D and G ...
            if sample_interval > 0 and step > 0 and step % sample_interval == 0:
                sample(G_AB, realA, step, output_dir)
                if args.verbose:
                    tqdm.write(f"💾 Saved sample at step {step}")

            if checkpoint_interval > 0 and step > 0 and step % checkpoint_interval == 0:
                save_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, output_dir, metrics)
                if args.verbose:
                    tqdm.write(f"✅ Saved checkpoint at step {step}")
            step += 1
        # ... end of epoch logic ...
        if avg_g_loss < metrics['best_g_loss']:
            metrics['best_g_loss'] = avg_g_loss
            torch.save(G_AB.state_dict(), f"{output_dir}/G_A2B_best.pth")
            torch.save(G_BA.state_dict(), f"{output_dir}/G_B2A_best.pth")

    save_final_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, output_dir, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker Environment Arguments
    parser.add_argument('--hosts', type=list, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--checkpoint-dir', type=str, default='/opt/ml/checkpoints')
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--num-workers', type=int, default=4)

    # Your Script's Hyperparameters (passed from the launcher)
    parser.add_argument("--bs",   type=int, default=4, help="Batch size")
    parser.add_argument("--lr",   type=float, default=2e-4, help="Learning rate")
    parser.add_gument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lambda-cyc", type=float, default=10.0, help="Cycle consistency weight")
    parser.add_argument("--samples-per-epoch", type=int, default=2)
    parser.add_argument("--checkpoint-epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Filename of checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    train(args)
