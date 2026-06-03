import torch
import torch.nn as nn

import numpy as np
import argparse
import os
import logging

from data import get_dataloader
import utils
import plot

import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='vit_so400m_patch14_siglip_378.v2_webli', # this is the one we use but the dinov3 models are good too
                        choices=['vit_base_patch16_dinov3.lvd1689m', 
                                'vit_large_patch16_dinov3.lvd1689m',
                                'vit_huge_plus_patch16_dinov3.lvd1689m',
                                'vit_7b_patch16_dinov3.lvd1689m', 
                                'vit_so400m_patch14_siglip_378.v2_webli']) 
    parser.add_argument('--use_routing', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1666)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.025)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'])
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--use_cutmix', action='store_true', default=False)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)

    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.5)


    args = parser.parse_args()

    utils.set_seed(args.seed)
    model_name = utils.get_model_name(args)

    run_dir = os.path.join(
        'Experiments',
        str(args.seed),
        model_name
    )
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    log_path = os.path.join(run_dir, 'train.log')

    print(f'Run dir : {run_dir}')

    device = utils.get_pytorch_device()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'Device  : {device}  |  GPUs: {n_gpus}')

    dset = get_dataloader(args.batch_size, args.num_workers, args.image_size, args.folds)

    # reference code https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/criterions/label_smoothed_cross_entropy.html
    def smooth_nll_loss(log_probs, targets, smoothing=0.1, num_classes=100):
        confidence = 1.0 - smoothing
        smooth_val = smoothing / (num_classes - 1)

        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.sum(dim=1)

        return (confidence * nll_loss + smooth_val * smooth_loss).mean()
    
    criterion = smooth_nll_loss if args.use_routing else nn.CrossEntropyLoss(label_smoothing=0.1)

    fold_results = {}
    fold_metrics = []
    for fold in range(args.folds):
        model = utils.build_model(args, device)

        best_acc, metrics = train.run_fold(
            args, start_epoch=0,
            model=model, criterion=criterion,
            dataset=dset, device=device,
            fold=fold,
        )
        fold_results[fold] = best_acc
        fold_metrics.append(metrics)

    print(f'\nAll folds complete. Per-fold best: {fold_results}')
    print(f'Mean: {np.mean(list(fold_results.values())):.2f}%  '
                f'Std: {np.std(list(fold_results.values())):.2f}%')
    
    plot.plot(fold_metrics, run_dir)
