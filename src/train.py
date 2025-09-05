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
    """Saves a grid of real and fake images. This is a final artifact."""
    fakeB = generator(batchA).detach().cpu()*0.5+0.5
    realA = batchA.detach().cpu()*0.5+0.5
    grid = torch.cat([realA, fakeB], dim=0)
    # Sample images are saved to the final model directory, not the checkpoint directory
    save_image(grid, os.path.join(out_dir, f"sample_{step:06}.jpg"), nrow=len(realA))

def save_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, checkpoint_dir, metrics):
    """Save a training checkpoint to the directory specified by SageMaker for checkpointing."""
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    # Save the full checkpoint file that can be used for resuming
    checkpoint_filename = f"checkpoint_step_{step}.pth"
    torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_filename))

def save_final_models(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, out_dir, metrics):
    """Save final models and training history to the main model output directory."""
    # This function saves the final artifacts to be packaged into model.tar.gz
    # Save complete checkpoint for archival purposes
    final_checkpoint = {
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
    torch.save(final_checkpoint, os.path.join(out_dir, "checkpoint_final.pth"))

    # Save just the final generator weights for easy inference
    torch.save(G_AB.state_dict(), os.path.join(out_dir, "G_A2B_final.pth"))
    torch.save(G_BA.state_dict(), os.path.join(out_dir, "G_B2A_final.pth"))

    # Save training history as a JSON file
    with open(os.path.join(out_dir, "training_history.json"), 'w') as f:
        json.dump({
            'g_losses': metrics['g_losses'],
            'd_losses': metrics['d_losses'],
            'best_g_loss': metrics['best_g_loss'],
            'total_steps': step,
            'total_epochs': epoch + 1
        }, f, indent=2)

def load_checkpoint(checkpoint_path, G_AB, G_BA, D_A, D_B, g_opt, d_opt):
    """Resume training from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    D_A.load_state_dict(checkpoint['D_A'])
    D_B.load_state_dict(checkpoint['D_B'])
    g_opt.load_state_dict(checkpoint['g_opt'])
    d_opt.load_state_dict(checkpoint['d_opt'])
    print(f"Resumed from epoch {checkpoint['epoch']} at step {checkpoint['step']}")
    # Return metrics if they exist in the checkpoint, otherwise return an empty dict
    return checkpoint['epoch'], checkpoint['step'], checkpoint.get('metrics', {})

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory for final model artifacts (/opt/ml/model)
    output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # Directory for intermediate checkpoints (/opt/ml/checkpoints)
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Networks and Optimizers
    G_AB, G_BA = Generator().to(device), Generator().to(device)
    D_A, D_B = Discriminator().to(device), Discriminator().to(device)
    g_opt = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))

    # Loss Functions
    mae = nn.L1Loss()
    mse = nn.MSELoss()

    # --- Resume from Checkpoint Logic ---
    start_epoch, step = 0, 0
    metrics = {'g_losses': [], 'd_losses': [], 'best_g_loss': float('inf')}

    if args.resume:
        # SageMaker automatically downloads checkpoint data to args.checkpoint_dir
        resume_file_path = os.path.join(checkpoint_dir, args.resume)
        if os.path.exists(resume_file_path):
            start_epoch, step, metrics = load_checkpoint(resume_file_path, G_AB, G_BA, D_A, D_B, g_opt, d_opt)
        else:
            print(f"WARNING: Checkpoint file '{args.resume}' not found in {checkpoint_dir}. Starting from scratch.")

    # Datasets and Dataloaders
    A = ImageFolder(args.data_dir, domain="trainA")
    B = ImageFolder(args.data_dir, domain="trainB")
    loaderA = DataLoader(A, args.bs, True, drop_last=True, num_workers=args.num_workers)
    loaderB = DataLoader(B, args.bs, True, drop_last=True, num_workers=args.num_workers)

    # Dynamic calculation of intervals
    steps_per_epoch = min(len(A)//args.bs, len(B)//args.bs)
    sample_interval = max(1, steps_per_epoch // args.samples_per_epoch) if args.samples_per_epoch > 0 else 0
    checkpoint_interval = steps_per_epoch * args.checkpoint_epochs if args.checkpoint_epochs > 0 else 0

    print(f"Training on {device.upper()}.")
    print(f"Final models will be saved to: {output_dir}")
    print(f"Intermediate checkpoints will be saved to: {checkpoint_dir}")

    for epoch in range(start_epoch, args.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(zip(loaderA, loaderB), desc=f"Epoch {epoch+1}/{args.epochs}", total=steps_per_epoch)

        for realA, realB in pbar:
            realA, realB = realA.to(device), realB.to(device)

            # --- Train Generators ---
            g_opt.zero_grad()
            fakeB = G_AB(realA)
            fakeA = G_BA(realB)
            recA = G_BA(fakeB)
            recB = G_AB(fakeA)
            id_A = G_BA(realA)
            id_B = G_AB(realB)
            loss_id = (mae(id_A, realA) + mae(id_B, realB)) * 0.5
            loss_gan_BA = mse(D_A(fakeA), torch.ones_like(D_A(fakeA)))
            loss_gan_AB = mse(D_B(fakeB), torch.ones_like(D_B(fakeB)))
            loss_gan = loss_gan_AB + loss_gan_BA
            loss_cyc = mae(recA, realA) + mae(recB, realB)
            g_loss = loss_gan + args.lambda_cyc * loss_cyc + (args.lambda_cyc * 0.5) * loss_id
            g_loss.backward()
            g_opt.step()

            # --- Train Discriminators ---
            d_opt.zero_grad()
            loss_D_A = (mse(D_A(realA), torch.ones_like(D_A(realA))) + mse(D_A(fakeA.detach()), torch.zeros_like(D_A(fakeA.detach())))) * 0.5
            loss_D_A.backward()
            loss_D_B = (mse(D_B(realB), torch.ones_like(D_B(realB))) + mse(D_B(fakeB.detach()), torch.zeros_like(D_B(fakeB.detach())))) * 0.5
            loss_D_B.backward()
            d_opt.step()
            d_loss = loss_D_A + loss_D_B

            # Accumulate and display losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix(G_loss=f"{g_loss.item():.3f}", D_loss=f"{d_loss.item():.3f}")

            # --- Sampling and Checkpointing ---
            # Sample images are final artifacts, save to output_dir
            if sample_interval > 0 and step > 0 and step % sample_interval == 0:
                sample(G_AB, realA, step, output_dir)
                if args.verbose:
                    tqdm.write(f"💾 Saved sample image at step {step} to {output_dir}")

            # Intermediate checkpoints are saved to checkpoint_dir for S3 sync
            if checkpoint_interval > 0 and step > 0 and step % checkpoint_interval == 0:
                save_checkpoint(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, checkpoint_dir, metrics)
                if args.verbose:
                    tqdm.write(f"✅ Saved checkpoint at step {step} to {checkpoint_dir}")

            step += 1

        # --- End of Epoch Logic ---
        avg_g_loss = epoch_g_loss / steps_per_epoch
        avg_d_loss = epoch_d_loss / steps_per_epoch
        metrics['g_losses'].append(avg_g_loss)
        metrics['d_losses'].append(avg_d_loss)
        print(f"End of Epoch {epoch+1}/{args.epochs} -> Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")

        # Best models are final artifacts, save to output_dir
        if avg_g_loss < metrics['best_g_loss']:
            metrics['best_g_loss'] = avg_g_loss
            torch.save(G_AB.state_dict(), os.path.join(output_dir, "G_A2B_best.pth"))
            torch.save(G_BA.state_dict(), os.path.join(output_dir, "G_B2A_best.pth"))
            if args.verbose:
                print(f"✨ New best generator loss: {avg_g_loss:.4f}. Saved best models to {output_dir}.")

    # Final models are the primary output, save to output_dir
    save_final_models(G_AB, G_BA, D_A, D_B, g_opt, d_opt, epoch, step, output_dir, metrics)
    print(f"Training complete. Final models and artifacts saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker Environment Arguments
    parser.add_argument('--hosts', type=list, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'), help="Directory for final model artifacts, provided by SageMaker.")
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'), help="Directory for input data, provided by SageMaker.")
    parser.add_argument('--checkpoint-dir', type=str, default=os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'), help="Directory for intermediate checkpoints, provided by SageMaker.")
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--num-workers', type=int, default=4)

    # Your Script's Hyperparameters
    parser.add_argument("--bs",   type=int, default=4, help="Batch size")
    parser.add_argument("--lr",   type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lambda-cyc", type=float, default=10.0, help="Cycle consistency weight")
    parser.add_argument("--samples-per-epoch", type=int, default=2, help="Number of sample images to save per epoch.")
    parser.add_argument("--checkpoint-epochs", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume", type=str, default=None, help="Filename of checkpoint (e.g., 'checkpoint_step_10000.pth') to resume from.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    train(args)
