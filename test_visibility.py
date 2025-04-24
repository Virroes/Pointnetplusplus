import argparse
import os
import logging
from pathlib import Path
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from data_utils.visibility_dataset import VisibilityDataset
from models.visibility_model import VisibilityPointNet


def parse_args():
    parser = argparse.ArgumentParser(description="Visibility Prediction Evaluation over Multiple Scenes")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to directory of .xyz/.npy files or single file for testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint .pth file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader')
    parser.add_argument('--num_point', type=int, default=6144,
                        help='Number of points per angular patch')
    parser.add_argument('--ang_radius', type=float, nargs='+', default=[0.1, 0.5],
                        help='One or more angular radii (radians) for grouping')
    parser.add_argument('--sample_rate', type=float, default=1.5,
                        help='Sampling rate (fraction of total patches)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device id')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory name for logs (else timestamp)')
    return parser.parse_args()


def setup_logger(log_file: Path):
    logger = logging.getLogger('VisibilityEval')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Prepare logging
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    log_base = Path('./logs'); log_base.mkdir(exist_ok=True)
    run_dir = log_base / (args.log_dir or now); run_dir.mkdir(exist_ok=True)
    log_file = run_dir / 'evaluation.log'
    logger = setup_logger(log_file)
    logger.info('Started evaluation')
    logger.info(f'Arguments: {args}')

    # Discover test files
    data_path = Path(args.data)
    if data_path.is_dir():
        files = sorted(data_path.glob('*.xyz')) + sorted(data_path.glob('*.npy'))
    else:
        files = [data_path]
    logger.info(f'Found {len(files)} test scene files')

    # Build datasets and loader
    datasets = []
    for f in files:
        ds = VisibilityDataset(str(f),
                               num_point=args.num_point,
                               ang_radius=args.ang_radius[0],
                               sample_rate=args.sample_rate)
        datasets.append(ds)
        logger.info(f'  {f.name}: {len(ds)} samples')
    test_dataset = ConcatDataset(datasets)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    logger.info(f'Total test samples: {len(test_dataset)}, Batch size: {args.batch_size}')

    # Device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisibilityPointNet(num_classes=2).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Metrics
    total_samples = 0
    total_correct = 0
    conf = np.zeros((2,2), dtype=np.int64)

    # Evaluation loop
    with torch.no_grad():
        for feats, theta, phi, labels in tqdm(test_loader, desc='Evaluating', ncols=80):
            feats = feats.permute(0,2,1).to(device)
            theta = theta.to(device)
            phi   = phi.to(device)
            labels= labels.to(device)

            outputs = model(feats, theta, phi)      # (B,2,N)
            preds = outputs.argmax(dim=1)           # (B,N)

            preds_flat  = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()

            total_correct += (preds_flat == labels_flat).sum()
            total_samples += labels_flat.size
            for t,p in zip(labels_flat, preds_flat):
                conf[t,p] += 1

    # Compute and log results
    acc = total_correct / total_samples if total_samples>0 else 0.0
    logger.info(f'Total samples: {total_samples}')
    logger.info(f'Overall Accuracy: {acc*100:.2f}%')
    logger.info('Confusion Matrix (rows=true, cols=pred):')
    logger.info(f"\n{conf}")

    # Per-class metrics
    for cls in [0,1]:
        TP = conf[cls,cls]
        FN = conf[cls,:].sum() - TP
        FP = conf[:,cls].sum() - TP
        precision = TP / (TP+FP) if (TP+FP)>0 else 0.0
        recall    = TP / (TP+FN) if (TP+FN)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        logger.info(f'Class {cls}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')

    logger.info('Evaluation complete')

if __name__ == '__main__':
    main()