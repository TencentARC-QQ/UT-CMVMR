import torch
import torch.nn.functional as F

def cal_distance(x: torch.Tensor, y: torch.Tensor, distance_type: str, mode: str) -> torch.Tensor:
    if distance_type == "L2":
        if mode == "v-a":
            ones = torch.ones(x.shape[0], 1, requires_grad = False)
            if torch.cuda.is_available():
                ones = ones.cuda()
            x_expand = x.repeat(y.shape[0], 1)
            y_expand = torch.kron(y, ones)
        else:
            ones = torch.ones(y.shape[0], 1, requires_grad = False)
            if torch.cuda.is_available():
                ones = ones.cuda()
            x_expand = torch.kron(x, ones)
            y_expand = y.repeat(x.shape[0], 1)

        dist_xy = F.pairwise_distance(x_expand, y_expand)
        dist_xy = dist_xy.reshape(x.shape[0], y.shape[0])

    elif distance_type == "COS":
        # Already L2 normalized, so this operation will obtain cosine similarity,
        # -1 indicates that larger value means larger angle between samples from x, y
        dist_xy = (-1) * torch.matmul(x, y.transpose(1, 0))




    return dist_xy # shape[x.shape[0], y.shape[0]]    

class Recall():
    def __init__(self):
        self.distance_type = "L2"

    def cal_recall(self, x_embed: torch.Tensor, x_embed_rho: torch.Tensor, y_embed: torch.Tensor, y_embed_rho: torch.Tensor,distance_type: str, k: int, mode: str, weight: float):
        self.distance_type = distance_type
        # x_embed = F.normalize(x_embed, p=2, dim=1)
        # y_embed = F.normalize(y_embed, p=2, dim=1)
        dist_xy = (-1) * cal_distance(x_embed, y_embed, self.distance_type, mode)
        rho_dist_xy = (-1) * cal_distance(x_embed_rho, y_embed_rho, self.distance_type, mode)
        dist_xy = weight * dist_xy + (1-weight) * rho_dist_xy
        if mode == "a-v":
            values, indexes = dist_xy.topk(k, dim = 1)
        elif mode == "v-a":
            values, indexes = dist_xy.topk(k, dim = 0)
        return indexes, values
