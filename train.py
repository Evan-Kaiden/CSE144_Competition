import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from routing_network import RoutingClassifier

import utils
def get_newly_trainable_params(module, already_tracked: set):
    """Return params with requires_grad=True that the optimizer doesn't know about yet."""
    new_params = []
    for p in module.parameters():
        if p.requires_grad and id(p) not in already_tracked:
            new_params.append(p)
    return new_params

def within_group_permutation(y, group_size=25):
    device = y.device
    groups = y // group_size
    perm = torch.arange(len(y), device=device)
    for g in torch.unique(groups).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            perm[idx] = idx[torch.randperm(len(idx), device=device)]
    return perm

def unwrap_model(model):
    is_parallel = isinstance(model, nn.DataParallel)
    inner = model.module if is_parallel else model
    return inner, is_parallel

def train_epoch(args, model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(train_loader, leave=False):
        x, y = x.to(device), y.to(device)

        B = x.shape[0]
        optimizer.zero_grad()

        if args.use_cutmix and args.beta > 0 and np.random.rand() < args.cutmix_prob:
            lam = np.random.beta(args.beta, args.beta)

            if args.use_routing:
                rand_index = within_group_permutation(y) # we should only apply cutmix and mixup with images of the same grouping for the router network
            else:
                rand_index = torch.randperm(B, device=device)

            target_a = y
            target_b = y[rand_index]

            bbx1, bby1, bbx2, bby2 = utils.rand_bbox(x.size(), lam)

            x_mixed = x.clone()
            x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

            output = model(x_mixed)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

        elif args.use_mixup and np.random.rand() < args.mixup_prob:
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            if args.use_routing:
                rand_index = within_group_permutation(y) # we should only apply cutmix and mixup with images of the same grouping for the router network
            else:
                rand_index = torch.randperm(B, device=device)

            target_a = y
            target_b = y[rand_index]

            x_mixed = lam * x + (1 - lam) * x[rand_index]

            output = model(x_mixed)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

        else:
            output = model(x)
            loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
        total += B

    train_loss = total_loss / len(train_loader)
    train_acc  = 100. * correct / total
    print(f'  [train] epoch {epoch:03d}  loss {train_loss:.4f}  acc {train_acc:.2f}%')
    return train_loss, train_acc


def test_epoch(model, test_loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    num_groups = 4
    group_correct       = torch.zeros(num_groups)
    group_total         = torch.zeros(num_groups)
    specialist_correct = torch.zeros(num_groups)
    specialist_total   = torch.zeros(num_groups)
    router_conf_sum     = 0.0
    is_routing_model    = False

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            B = x.shape[0]

            inner = model.module if isinstance(model, nn.DataParallel) else model
            if isinstance(inner, RoutingClassifier):
                is_routing_model = True
                logits, group_log_probs = model(x, return_group_probs=True)
            else:
                logits = model(x)
                group_log_probs = None

            loss = criterion(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += B

            if group_log_probs is not None:
                group_probs  = group_log_probs.exp()              # [B, num_groups]
                picks        = group_probs.argmax(dim=-1).cpu()   # [B]
                true_groups  = (y // 25).cpu()                    # [B]
                
                for g in range(num_groups):
                    mask = (true_groups == g)
                    group_total[g]   += mask.sum().float()
                    group_correct[g] += ((picks == true_groups) & mask).sum().float()

                router_conf_sum += group_probs.max(dim=-1).values.sum().item()

                preds = logits.argmax(1).cpu()
                for g in range(num_groups):
                    mask = (true_groups == g)
                    specialist_total[g]   += mask.sum().float()
                    specialist_correct[g] += (preds[mask] == y.cpu()[mask]).sum().float()

    test_loss = total_loss / len(test_loader)
    test_acc  = 100. * correct / total
    print(f'  [test]  epoch {epoch:03d}  loss {test_loss:.4f}  acc {test_acc:.2f}%')

    if is_routing_model:
        overall_router_acc = 100. * group_correct.sum() / group_total.sum()
        avg_confidence  = router_conf_sum / total

        print(f'  [router] overall acc       : {overall_router_acc:.2f}%')
        print(f'  [router] avg confidence    : {avg_confidence:.3f}')

        per_group_cls_acc = specialist_correct / specialist_total.clamp(min=1)
        print(f'  [specialist] per-group acc : {per_group_cls_acc.numpy().round(3)}')
        for g in range(num_groups):
            group_name = ['food', 'cars', 'planes', 'flowers'][g]
            print(f'  [specialist] group {g} ({group_name:>7s}): {100*per_group_cls_acc[g]:.2f}%  ({int(specialist_correct[g])}/{int(specialist_total[g])})')

    return test_loss, test_acc

def run_fold(args, start_epoch, model, criterion, dataset, device, fold):
    metrics = {"train_acc":[], "train_loss":[], "test_acc":[], "test_loss":[]}
    best_acc = 0.0
    best_ckpt = os.path.join(args.run_dir, f'best_model_fold_{fold}.pth')
    optimizer, scheduler = utils.init_optimizer_and_scheduler(args, model)
    dataset.set_fold(fold)


    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f'\n--- Fold {fold} | Epoch {epoch}/{args.epochs} ---')

        train_loss, train_acc = train_epoch(
            args, model, dataset.train_loader,
            optimizer, criterion, device, epoch,
        )
        test_loss, test_acc = test_epoch(
            model, dataset.test_loader,
            criterion, device, epoch,
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f'  train loss {train_loss:.4f}  acc {train_acc:.2f}%'
            f' | test loss {test_loss:.4f}  acc {test_acc:.2f}%'
        )

        metrics['train_acc'].append(train_acc)
        metrics['train_loss'].append(train_loss)
        metrics['test_acc'].append(test_acc)
        metrics['test_loss'].append(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    'epoch':     epoch,
                    'model':     m.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_acc':  best_acc,
                    'fold':      fold,
                    'args':      vars(args),
                },
                best_ckpt,
            )
            print(f'  ** new best {best_acc:.2f}% — saved to {best_ckpt} **')

    print(f'\nFold {fold} complete. Best test acc: {best_acc:.2f}%')
    return best_acc, metrics