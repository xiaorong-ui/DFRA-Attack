import os
import sys
import torch
import numpy as np
from torch import nn, Tensor
from abc import abstractmethod
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from typing import List, Any, Dict
from contextlib import contextmanager
import random

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_cluster_center(embedding_img: Tensor, num_cluster=5) -> Tensor:
    with suppress_output():
        _, cluster_center = kmeans(
            X=embedding_img,
            num_clusters=num_cluster,
            distance="euclidean",
            device=embedding_img.device,
            iter_limit=50
        )
    return cluster_center

def OT(src_dis, tgt_dis):
    src_dis_norm = F.normalize(src_dis, dim=1)
    tgt_dis_norm = F.normalize(tgt_dis, dim=1)
    sim = torch.einsum("md,nd->mn", src_dis_norm, tgt_dis_norm).contiguous()
    wdist = 1 - sim
    xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
    yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
    with torch.no_grad():
        KK = torch.exp(-wdist / 0.1)
        T = Sinkhorn(KK, xx, yy)
    if torch.isnan(T).any():
        return None
    sim_op = torch.sum(T * sim, dim=(0, 1))
    loss = torch.sum(sim_op)
    return loss

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for _ in range(100):
        r0 = r
        r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
        c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.outer(r, c) * K
    return T

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def global_local_attention_features(self, x: Tensor):
        """
        通用的 CLIP attention 提取接口。

        为什么把这段逻辑放在 Base 里：
        1. 你当前 attack 用到的 B16 / B32 / Laion / L336 都是 CLIP 风格视觉编码器；
        2. 它们都具备 `self.normalizer` 和 `self.model.vision_model` 这两个共同接口；
        3. 因此可以在这里统一提供一个“取 global / local / local-attention”的默认实现，
           避免在每个 extractor 里重复写一遍几乎相同的代码。

        返回值说明：
        - global_feature: [B, d]，CLS / global token 的归一化特征；
        - local_feature:  [B, N, d]，去掉 CLS 后的 patch token 特征；
        - local_attention:[B, N, N]，最后一层 self-attention，先对多头求均值，再只保留 local->local 部分。

        这里故意只取“最后一层 + head mean”：
        - 最后一层的关系语义最接近最终判别空间；
        - 先平均 head，可以显著降低损失噪声和显存/实现复杂度；
        - 对你现在的 attack 来说，这是一版更稳妥的 attention alignment 起点。
        """
        if not hasattr(self, "normalizer") or not hasattr(self, "model"):
            raise NotImplementedError(
                f"{self.__class__.__name__} 缺少 normalizer/model，无法走通用 attention 提取路径。"
            )
        if not hasattr(self.model, "vision_model"):
            raise NotImplementedError(
                f"{self.__class__.__name__} 不包含 vision_model，无法提取视觉 attention。"
            )

        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
        )
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=-1, keepdim=True)

        # 注意这里取的是最后一层 attention：
        # outputs.attentions[-1] 形状为 [B, num_heads, num_tokens, num_tokens]。
        # 我们对 head 求均值，再裁掉 CLS，只保留 patch token 之间的关系矩阵 [B, N, N]。
        local_attention = outputs.attentions[-1].mean(dim=1)[:, 1:, 1:]
        return global_feature, local_feature, local_attention

class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x)
        return features

class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor]) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss

class EnsembleExtractor_global(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleExtractor_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor):
        features = []
        for model in self.extractors:
            global_feature = model.global_features(x)
            features.append(global_feature.squeeze())
        return features,

class EnsembleLoss_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleLoss_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.previous_loss_list = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            x_tensor = model.global_features(x)
            self.ground_truth.append(x_tensor)

    def __call__(self, features: List[Tensor]):
        loss_list = []
        for index in range(len(self.extractors)):
            gt = self.ground_truth[index]
            feature = features[index].unsqueeze(0)

            feat_loss = OT(gt, feature)
            loss_list.append(feat_loss)

        total_losses = [loss_list[i] for i in range(len(self.extractors))]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureExtractor_ot(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureExtractor_ot, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor):
        features = []
        features_local = []
        features_attention = []
        for model in self.extractors:
            # 这里把 attention 一起取出来，是为了把“区域间谁关注谁”的关系
            # 也纳入对齐目标，而不仅仅是 patch feature 本身的 Gram 关系。
            global_feature, local_feature, local_attention = model.global_local_attention_features(x)
            features.append(global_feature.squeeze())
            # 🔥 1. 注释掉 K-means 聚类
            # cluster_center = get_cluster_center(local_feature[0], self.cluster_number).unsqueeze(0)
            # features_local.append(cluster_center)
            # 🔥 2. 直接将完整的 196 个 Token 存入列表
            features_local.append(local_feature[0].unsqueeze(0))
            features_attention.append(local_attention[0].unsqueeze(0))
        return features, features_local, features_attention

class EnsembleFeatureLoss_OT(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, features: List[Tensor], features_local: List[Tensor]):
        loss = 0
        loss_local=0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = features[index]
            feature_local = features_local[index].squeeze(0)
            loss_local += OT(gt_local, feature_local) * 2
            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)
        loss_local = loss_local / len(self.extractors)

        loss = loss + loss_local * 0.1
        return loss

class EnsembleFeatureLoss_OT_Auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT_Auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, features: List[Tensor], features_local: List[Tensor]):
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = features[index]
            feature_local = features_local[index].squeeze(0)
            local_loss = OT(gt_local, feature_local) * 2
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.1 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K    

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss


def Sinkhorn(K, u, v):
    # 这里给 Sinkhorn 补上数值稳定性保护。
    # 之前直接做除法：
    #   r = u / (K @ c)
    #   c = v / (K^T @ r)
    # 当某一行/列的和非常接近 0 时，容易产生 inf / nan；
    # CUDA 往往不会在出问题的那一行立刻报错，而是异步地在后续某个 `.item()` /
    # matmul / reduction 位置抛出 "unspecified launch failure"。
    # 因此这里统一：
    # 1) 分母 clamp_min；
    # 2) 迭代中检查有限性；
    # 3) 若 transport plan 非法，返回 None 让上层兜底。
    r = torch.ones_like(u, dtype=torch.float32)
    c = torch.ones_like(v, dtype=torch.float32)
    K = K.to(dtype=torch.float32)
    u = u.to(dtype=torch.float32)
    v = v.to(dtype=torch.float32)

    thresh = 1e-2
    eps = 1e-8
    for _ in range(100):
        r0 = r

        denom_r = (K @ c.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        r = u / denom_r

        denom_c = (K.t() @ r.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        c = v / denom_c

        if not torch.isfinite(r).all() or not torch.isfinite(c).all():
            return None

        err = (r - r0).abs().mean()
        if not torch.isfinite(err):
            return None
        if err.item() < thresh:
            break

    T = torch.outer(r, c) * K
    if not torch.isfinite(T).all():
        return None
    return T

def OT(src_dis, tgt_dis):
    # 这里的 OT 既要给 attack 提供可反传的相似性分数，又不能因为极端输入把整轮攻击炸掉。
    # 因此做几层保护：
    # 1) 空 tensor 直接返回 0；
    # 2) 内部统一用 float32 计算 transport plan，降低数值抖动；
    # 3) 只让 `sim` 参与梯度，Sinkhorn plan `T` 仍然保持 no_grad；
    # 4) 一旦发现非有限值，就返回 0 分而不是把整个 job 打崩。
    if src_dis is None or tgt_dis is None:
        device = None
        if isinstance(src_dis, torch.Tensor):
            device = src_dis.device
        elif isinstance(tgt_dis, torch.Tensor):
            device = tgt_dis.device
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    if src_dis.numel() == 0 or tgt_dis.numel() == 0:
        return src_dis.new_tensor(0.0)

    if src_dis.dim() != 2 or tgt_dis.dim() != 2:
        raise ValueError(
            f"OT expects 2D tensors, but got {tuple(src_dis.shape)} and {tuple(tgt_dis.shape)}"
        )

    src_dis_norm = F.normalize(src_dis.float(), dim=1)
    tgt_dis_norm = F.normalize(tgt_dis.float(), dim=1)
    sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()

    if not torch.isfinite(sim).all():
        return sim.new_tensor(0.0)

    wdist = 1 - sim.detach()
    xx = torch.full(
        (src_dis.shape[0],),
        1.0 / max(src_dis.shape[0], 1),
        dtype=torch.float32,
        device=sim.device,
    )
    yy = torch.full(
        (tgt_dis.shape[0],),
        1.0 / max(tgt_dis.shape[0], 1),
        dtype=torch.float32,
        device=sim.device,
    )

    with torch.no_grad():
        KK = torch.exp(-wdist / 0.1)
        if not torch.isfinite(KK).all():
            return sim.new_tensor(0.0)
        T = Sinkhorn(KK, xx, yy)

    if T is None:
        return sim.new_tensor(0.0)

    T = T.to(device=sim.device, dtype=sim.dtype)
    sim_op = torch.sum(T * sim, dim=(0, 1))
    loss = torch.sum(sim_op)

    if not torch.isfinite(loss):
        return sim.new_tensor(0.0)
    return loss





# ========================================================
class EnsembleFeatureExtractor_middle(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureExtractor_middle, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}

        features_global_middle = {}
        features_local_middle = {}
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding, middle_tensor, middle_embedding = model.global_local_middle_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = get_cluster_center(x_embedding[0], self.cluster_number).unsqueeze(0)
            features_local[i] = cluster_center

            features_global_middle[i] = middle_tensor.squeeze()
            cluster_middle_center = get_cluster_center(middle_embedding[0], self.cluster_number).unsqueeze(0)
            features_local_middle[i] = cluster_middle_center

        return features, features_local, features_global_middle, features_local_middle

class EnsembleFeatureLoss_OT_dfra_attack_middle(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5):
        super(EnsembleFeatureLoss_OT_dfra_attack_middle, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number
        self.ground_truth_middle = []
        self.ground_truth_local_middle = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_middle.clear()
        self.ground_truth_local_middle.clear()
        for model in self.extractors:
            x_tensor, x_embedding, x_tensor_middle, cluster_center_middle = model.global_local_middle_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding, self.cluster_number).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

            self.ground_truth_middle.append(x_tensor_middle)
            self.ground_truth_local_middle.append(cluster_center_middle)

    def __call__(
        self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor],
        feature_middle_dict: Dict[int, Tensor], feature_local_middle_dict: Dict[int, Tensor]
    ):
        loss_list = []
        loss_local_list = []
        loss_local_middle_list= []
        loss_middle_list= []

        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt_local_middle = self.ground_truth_local_middle[index].squeeze(0)
            gt = self.ground_truth[index]
            gt_middle = self.ground_truth_middle[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)
            feature_middle = feature_middle_dict[index].unsqueeze(0)
            feature_local_middle = feature_local_middle_dict[index].squeeze(0)

            local_loss = OT(gt_local, feature_local)
            feat_loss = OT(gt, feature)
            local_middle_loss = OT(gt_local_middle, feature_local_middle)
            feat_middle_loss = OT(gt_middle, feature_middle)
            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)
            loss_local_middle_list.append(local_middle_loss)
            loss_middle_list.append(feat_middle_loss)

        total_losses = [
            loss_list[i]
            + 0.2 * loss_local_list[i]
            + 0.2 * loss_middle_list[i]
            + 0.2 * loss_local_middle_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureExtractor_middle_non_local(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureExtractor_middle_non_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_global_random = {}

        for i, model in enumerate(self.extractors):
            x_tensor, index_tensor = model.global_middle_features(x.to(x.device))
            features[i] = x_tensor.squeeze()

            features_global_random[i] = index_tensor.squeeze()

        return features, features_global_random

class EnsembleFeatureLoss_OT_dfra_attack_middle_non_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_dfra_attack_middle_non_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number
        self.ground_truth_middle = []
        self.ground_truth_local_middle = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_middle.clear()
        self.ground_truth_local_middle.clear()
        for model in self.extractors:
            x_tensor, x_tensor_middle= model.global_middle_features(x.to(x.device))
            self.ground_truth.append(x_tensor)

            self.ground_truth_middle.append(x_tensor_middle)

    def __call__(self, feature_dict: Dict[int, Tensor], feature_middle_dict: Dict[int, Tensor]):
        loss_list = []
        loss_middle_list= []
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            gt_middle = self.ground_truth_middle[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_middle = feature_middle_dict[index].unsqueeze(0)

            feat_loss = OT(gt, feature)
            feat_middle_loss = OT(gt_middle, feature_middle)
            loss_list.append(feat_loss)
            loss_middle_list.append(feat_middle_loss)

        total_losses = [
            loss_list[i]
            + 0.2 * loss_middle_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

import random

class EnsembleFeatureExtractor_random(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],
        random_layer: List[int], random_index: List[float],
        index_in_layer: bool=True, cluster_number=5
    ):
        super(EnsembleFeatureExtractor_random, self).__init__()
        self.extractors = nn.ModuleList(extractors)

        self.random_layer = random_layer
        self.random_index = random_index
        self.index_in_layer = index_in_layer
        self.cluster_number = cluster_number

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    def forward(self, x: Tensor):
        features = []
        features_local = []
        features_global_random = []
        features_local_random = []

        for i, model in enumerate(self.extractors):

            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            global_last, local_last, global_i, local_i = model.global_local_index_features(x, index)

            features.append(global_last.squeeze())
            cluster_center = get_cluster_center(local_last[0], self.cluster_number).unsqueeze(0)
            features_local.append(cluster_center)

            features_global_random.append(global_i.squeeze())
            cluster_random_center = get_cluster_center(local_i[0], self.cluster_number).unsqueeze(0)
            features_local_random.append(cluster_random_center)

        return features, features_local, features_global_random, features_local_random

class EnsembleFeatureLoss_OT_dfra_attack_random(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],
        random_layer: List[int], random_index: List[float],
        pin_layer: bool=False, pin_index: bool=False, index_in_layer: bool=True,
        cluster_number=5
    ):
        super(EnsembleFeatureLoss_OT_dfra_attack_random, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.ground_truth_random = []
        self.ground_truth_local_random = []

        self.random_layer = random_layer
        self.random_index = random_index
        self.pin_layer = pin_layer
        self.pin_index = pin_index
        self.index_in_layer = index_in_layer
        self.cluster_number = cluster_number

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_random.clear()
        self.ground_truth_local_random.clear()

        if not self.pin_layer:
            random.shuffle(self.random_layer)

        for i, model in enumerate(self.extractors):

            if not self.pin_index:
                self.random_index[i] = random.random()
            
            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            x_tensor, x_embedding, x_tensor_i, cluster_center_i = model.global_local_index_features(x.to(x.device), index)
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding, self.cluster_number).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

            self.ground_truth_random.append(x_tensor_i)
            self.ground_truth_local_random.append(cluster_center_i)

    def __call__(self,
        features: List[Tensor],
        features_local: List[Tensor],
        features_random: List[Tensor],
        features_local_random: List[Tensor]
    ) -> Tensor:
        loss_list = []
        loss_local_list = []
        loss_local_middle_list= []
        loss_middle_list= []

        for index, model in enumerate(self.extractors):

            gt_local = self.ground_truth_local[index].squeeze(0)
            gt_local_middle = self.ground_truth_local_random[index].squeeze(0)
            gt = self.ground_truth[index]
            gt_middle = self.ground_truth_random[index]

            feature = features[index].unsqueeze(0)
            feature_local = features_local[index].squeeze(0)
            feature_random = features_random[index].unsqueeze(0)
            feature_local_random = features_local_random[index].squeeze(0)

            local_loss = OT(gt_local, feature_local)
            feat_loss = OT(gt,feature)
            local_middle_loss = OT(gt_local_middle, feature_local_random)
            feat_middle_loss = OT(gt_middle, feature_random)
            
            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)
            loss_local_middle_list.append(local_middle_loss)
            loss_middle_list.append(feat_middle_loss)

        total_losses = [
            loss_list[i]
            + 0.2 * loss_local_list[i]
            + 0.2 * loss_middle_list[i]
            + 0.2 * loss_local_middle_list[i]
            for i in range(len(self.extractors))
        ]

        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureExtractor_random_non_local(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],
        random_layer: List[int], random_index: List[float], index_in_layer: bool=True
    ):
        super(EnsembleFeatureExtractor_random_non_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)

        self.random_layer = random_layer
        self.random_index = random_index
        self.index_in_layer = index_in_layer

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_global_random = {}

        for i, model in enumerate(self.extractors):

            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            x_tensor, index_tensor = model.global_index_features(x.to(x.device), index)
            features[i] = x_tensor.squeeze()
            features_global_random[i] = index_tensor.squeeze()

        return features, features_global_random

class EnsembleFeatureExtractor_ot3(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_ot3, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        features_local = {}
        for i, model in enumerate(self.extractors):
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = get_cluster_center(x_embedding[0], 3).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

class EnsembleFeatureLoss_OT_dfra_attack_random_non_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor],
        random_layer: List[int], random_index: List[float],
        pin_layer: bool=False, pin_index: bool=False, index_in_layer: bool=True
    ):
        super(EnsembleFeatureLoss_OT_dfra_attack_random_non_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.ground_truth_random = []
        self.ground_truth_local_random = []

        self.random_layer = random_layer
        self.random_index = random_index
        self.pin_layer = pin_layer
        self.pin_index = pin_index
        self.index_in_layer = index_in_layer

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_random.clear()
        self.ground_truth_local_random.clear()
        
        if not self.pin_layer:
            random.shuffle(self.random_layer)

        for i, model in enumerate(self.extractors):

            if not self.pin_index:
                self.random_index[i] = random.random()
            
            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            x_tensor, x_tensor_i = model.global_index_features(x.to(x.device), index)
            self.ground_truth.append(x_tensor)
            self.ground_truth_random.append(x_tensor_i)

    def __call__(self, feature_dict: Dict[int, Tensor], feature_middle_dict: Dict[int, Tensor]):
        loss_list = []
        loss_middle_list= []

        for index, model in enumerate(self.extractors):

            gt = self.ground_truth[index]
            gt_middle = self.ground_truth_random[index]

            feature = feature_dict[index].unsqueeze(0)
            feature_middle = feature_middle_dict[index].unsqueeze(0)

            feat_loss = OT(gt, feature)
            feat_middle_loss = OT(gt_middle, feature_middle)
            
            loss_list.append(feat_loss)
            loss_middle_list.append(feat_middle_loss)

        total_losses = [
            loss_list[i]
            + 0.2 * loss_middle_list[i]
            for i in range(len(self.extractors))
        ]

        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureLoss_OT_ablation_wo_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding, self.cluster_number).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor]):
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = OT(gt_local, feature_local)
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureLoss_OT_ablation_wo_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding, self.cluster_number).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor]):
        loss_list = []
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)

            feat_loss = OT(gt, feature)
            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

class EnsembleFeatureLoss_OT_ablation_wo_dynamic(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_dynamic, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = get_cluster_center(x_embedding, self.cluster_number).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor]):
        loss_list = []
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)

            feat_loss = OT(gt, feature)
            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]

        total_loss = sum(total_losses)/len(total_losses)
        return total_loss
