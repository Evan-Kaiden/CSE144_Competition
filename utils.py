import torch
import numpy as np
import torch.nn as nn

def get_pytorch_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def set_seed(seed: int = 42):
    """Make results as reproducible as possible across runs."""
    import os, random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_model_name(args):
    model_name = ''
    model_name += args.backbone
    model_name += f'_batch_size:{args.batch_size}'
    model_name += f'_img_size:{args.image_size}'
    model_name += '_cutmix' if args.use_cutmix else ''
    model_name += '_mixup' if args.use_mixup else ''
    model_name += '_routing' if args.use_routing else ''
    model_name += '_lr:' + str(args.lr)

    return model_name

def init_optimizer_and_scheduler(args, model):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay)

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    else:
        scheduler = None

    return optimizer, scheduler



def build_model(args, device, ignore_parallel=False, load_weights=True):
    from routing_network import RoutingClassifier
    from torch.nn.parallel import DataParallel
    import timm

    if args.use_routing:
        backbone = timm.create_model(args.backbone, pretrained=load_weights, num_classes=0)
        feature_dim = backbone.num_features
        model = RoutingClassifier(backbone, feature_dim, num_groups=4, classes_per_group=25)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.router_head.parameters():
            p.requires_grad = True
        for p in model.specialist_heads.parameters():
            p.requires_grad = True
    else:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=100)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True

    if ignore_parallel:
        return model
    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        model = DataParallel(model)
    return model