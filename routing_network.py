import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int,
                 num_groups: int = 4, classes_per_group: int = 25):
        super().__init__()
        self.backbone = backbone
        self.num_groups = num_groups
        self.classes_per_group = classes_per_group

        self.router_head = nn.Linear(feature_dim, num_groups)
        self.specialist_heads = nn.ModuleList([
            nn.Linear(feature_dim, classes_per_group) for _ in range(num_groups)
        ])

    def forward(self, x, return_group_probs=False):

        features = self.backbone(x)
    
        group_log_probs = F.log_softmax(self.router_head(features), dim=-1)

        specialist_log_probs = torch.stack([F.log_softmax(head(features), dim=-1) for head in self.specialist_heads], dim=1)
        
        combined = group_log_probs.unsqueeze(-1) + specialist_log_probs
        if return_group_probs:
            return combined.view(combined.size(0), -1), group_log_probs
        return combined.view(combined.size(0), -1)