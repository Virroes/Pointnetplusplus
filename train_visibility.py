import argparse
import os
import logging
from pathlib import Path
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from data_utils.visibility_dataset import VisibilityDataset
from models.visibility_model import VisibilityPointNet


def parse_args():
    parser = argparse.ArgumentParser(description="Visibility Prediction Training over Multiple Scenes with Class Weights and Logging")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to directory of .xyz/.npy files, or single file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of patches per batch')
    parser.add_argument('--num_point', type=int, default=6144,
                        help='Points per angular patch')
    parser.add_argument('--ang_radius', type=float, nargs='+', default=[0.1, 0.5],
                        help='One or more angular radii (radians) for multi-scale grouping')
    parser.add_argument('--sample_rate', type=float, default=1.5,
                        help='Fraction of (total_points/num_point) patches per scene')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', choices=['AdamW','SGD'], default='AdamW',
                        help='Optimizer type')
    parser.add_argument('--lr_schedule', choices=['step','cosine'], default='cosine',
                        help='LR scheduling strategy')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Decay step size (if using step scheduler)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Decay factor (if using step scheduler)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device id')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory name for logs (else timestamp)')
    return parser.parse_args()


def setup_logger(log_file: Path):
    logger = logging.getLogger('VisibilityTraining')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup logging directory
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    log_base = Path('./logs'); log_base.mkdir(exist_ok=True)
    run_dir = log_base / (args.log_dir or now); run_dir.mkdir(exist_ok=True)
    log_file = run_dir / 'training.log'
    logger = setup_logger(log_file)
    logger.info('Started training run')
    logger.info(f'Arguments: {args}')

    # Discover scene files
    data_path = Path(args.data)
    if data_path.is_dir():
        files = sorted(data_path.glob('*.xyz')) + sorted(data_path.glob('*.npy'))
    else:
        files = [data_path]
    logger.info(f'Found {len(files)} scene files')

    # Build per-scene datasets
    datasets = []
    for f in files:
        ds = VisibilityDataset(str(f),
                               num_point=args.num_point,
                               ang_radius=args.ang_radius[0],
                               sample_rate=args.sample_rate)
        datasets.append(ds)
        logger.info(f'  {f.name}: {len(ds)} samples')
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    logger.info(f'Total training samples: {len(train_dataset)}, Batch size: {args.batch_size}')

    # Device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisibilityPointNet(num_classes=2).to(device)

    # Compute class weights across scenes
    labels_all = np.concatenate([
        (np.load(str(f)) if f.suffix=='.npy' else np.loadtxt(str(f)))[:,5].astype(np.int64)
        for f in files
    ])
    counts = np.bincount(labels_all, minlength=2)
    total = labels_all.size
    class_weights = torch.tensor([total/counts[0], total/counts[1]], dtype=torch.float32).to(device)
    logger.info(f'Class counts: {counts.tolist()}, Weights: {[float(w) for w in class_weights]}')

    # Loss & optimizer
    criterion = nn.NLLLoss(weight=class_weights).to(device)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler & warmup
    if args.lr_schedule=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    logger.info('Beginning training')
    for epoch in range(1, args.epochs+1):
        model.train(); epoch_loss=0; epoch_correct=0; epoch_samples=0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80)
        for feats, theta, phi, labels in loop:
            feats = feats.permute(0,2,1).to(device)
            theta = theta.to(device); phi = phi.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats, theta, phi).permute(0,2,1).reshape(-1,2)
            targets = labels.view(-1)
            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            preds = outputs.argmax(dim=1)
            correct = (preds==targets).sum().item(); num=targets.numel()
            epoch_loss+=loss.item()*num; epoch_correct+=correct; epoch_samples+=num
            loop.set_postfix(loss=epoch_loss/epoch_samples, acc=epoch_correct/epoch_samples)

        # Scheduler step after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()
        avg_loss=epoch_loss/epoch_samples; avg_acc=epoch_correct/epoch_samples
        lr_current = optimizer.param_groups[0]['lr']
        logger.info(f"End epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, LR={lr_current:.6f}")
        ckpt_path = run_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict()}, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info('Training complete')

if __name__=='__main__':
    main()
