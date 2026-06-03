import torch
import os


main_path = 'Experiments/1666'

models = [path for path in os.listdir(main_path) if 'lr' in path]

model_to_acc = {}

from tqdm import tqdm
for model in tqdm(models):
    ckpts = [path for path in os.listdir(os.path.join(main_path, model)) if '.pth' in path]

    accs = []
    for ckpt in ckpts:
        vals = torch.load(os.path.join(main_path, model, ckpt), weights_only=False, map_location='cpu')
        acc = vals['test_acc']
        accs.append(acc)

    model_to_acc[model] = sum(accs) / len(accs)


for model in models:
    print(f"{model}:{model_to_acc[model]}")