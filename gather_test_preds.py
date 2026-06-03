import torch
import pandas as pd
import argparse
import os
import glob
from argparse import Namespace
from PIL import Image
from tqdm import tqdm
import utils

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True,
                    help="Run folder (ensembles best_model_fold_*.pth inside) "
                         "or a single .pth file.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--output", type=str, default="res.csv")
args = parser.parse_args()

DEVICE = utils.get_pytorch_device()

if os.path.isdir(args.path):
    ckpt_paths = sorted(glob.glob(os.path.join(args.path, "best_model_fold_*.pth")))
    assert ckpt_paths, f"No best_model_fold_*.pth files in {args.path}"
else:
    ckpt_paths = [args.path]

print(f"Ensembling {len(ckpt_paths)} checkpoint(s):")
for p in ckpt_paths:
    print(f"  {p}")

# Normalize saved args to a Namespace (handles both vars(args) dict and Namespace saves)
def _to_ns(a):
    return a if hasattr(a, '__dict__') else Namespace(**a)

# Use the first checkpoint to set up the shared test transform
first = torch.load(ckpt_paths[0], weights_only=False, map_location='cpu')
first_args = _to_ns(first['args'])
image_size = first_args.image_size

mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]


class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.filenames = sorted(os.listdir(root))
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(os.path.join(self.root, fn)).convert('RGB')
        return self.transform(img), fn
    
base_transform = T.Compose([
    T.RandomResizedCrop(
        size=(image_size, image_size),
        scale=(0.8, 1.0),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

test_dataset = TestDataset('./data/test', base_transform)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=args.num_workers,
    pin_memory=(DEVICE == 'cuda'),
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=2,
)
N = len(test_dataset)
prob_sum = None

tta_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
])

for i, ckpt_path in enumerate(ckpt_paths):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)  
    saved_args = _to_ns(ckpt['args'])

    acc = ckpt.get('test_acc')
    print(f"\n[{i+1}/{len(ckpt_paths)}] {os.path.basename(ckpt_path)}  "
          f"epoch={ckpt.get('epoch','?')} fold={ckpt.get('fold','?')} acc={acc}")
    if acc is not None and acc < 90:
        print("acc is too low... skipping")
        continue

    model = utils.build_model(saved_args, DEVICE, ignore_parallel=True, load_weights=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()

    offset = 0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader):
            B = imgs.size(0)

            views = [imgs] + [tta_transform(imgs) for _ in range(5)]   # 6 × [B, C, H, W]
            imgs_stack = torch.cat(views, dim=0).to(DEVICE)            # [6B, C, H, W]

            logits = model(imgs_stack)                                  # single forward pass
            logits = logits.view(6, B, -1).mean(dim=0)                 # [B, num_classes]

            probs = logits.softmax(dim=-1)

            if prob_sum is None:
                prob_sum = torch.zeros(N, probs.shape[1], device=DEVICE)
            prob_sum[offset:offset+B] += probs
            offset += B

    del model
    torch.cuda.empty_cache()

# Average and predict
mean_probs = prob_sum / len(ckpt_paths)
preds = mean_probs.argmax(dim=-1).cpu().tolist()

df = pd.DataFrame({'ID': test_dataset.filenames,
                   'Label': [int(p) for p in preds]})
df.to_csv(args.output, index=False)
print(f"\nWrote {len(df)} predictions to {args.output}")