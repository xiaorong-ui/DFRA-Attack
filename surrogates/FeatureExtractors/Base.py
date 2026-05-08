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

# ========================================================
# 全局最优传输 (Optimal Transport) 距离计算函数
# ========================================================
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



import math
class EnsembleFeatureLoss_OT_dfra_attack(nn.Module):
    # 这里将 erasing_prob 保持为 1.0：
    # - 当前方法使用的是“shared top-k saliency mask”，不是普通随机擦除；
    # - 如果每个 step 只偶尔启用 mask，会让优化目标在 masked / unmasked 视图之间来回切换，
    #   不利于稳定地学习局部对齐。
    # 同时把默认 erasing_scale 调低到 0.12：
    # - 因为当前 mask 的是三个 CLIP 共同最关注的区域，破坏力比随机遮挡更强；
    # - 0.12 在 14x14 公共网格上大约对应 24 个 patch，通常比 0.15 更稳妥。
    def __init__(
        self,
        extractors,
        cluster_number=5,
        use_random_erasing=True,
        erasing_prob=1.0,
        erasing_scale=(0.20, 0.20),
        use_saliency_mask=True,
        lambda_attn=0.1,
        lambda_rel=0.5,
        use_mask=True,
    ):
        super(EnsembleFeatureLoss_OT_dfra_attack, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.ground_truth_attention = []
        self.previous_loss_list = []
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number

        # 掩码参数
        self.use_random_erasing = use_random_erasing if use_mask else False
        # self.use_random_erasing = False  # 消融实验，不使用mask试试
        self.erasing_prob = erasing_prob
        self.erasing_scale = erasing_scale
        self.use_saliency_mask = use_saliency_mask if use_mask else False
        self.use_mask = use_mask

        # 🔥 核心修正 0：新增对齐控制参数
        # current_mask_indices/current_mask_grid_shape 共同描述“当前 step 的共享 mask”：
        # 1) indices 存的是在公共空间网格上的 flat patch index；
        # 2) grid_shape 存的是公共空间网格尺寸，例如 224 输入 + patch_size=16 -> (14, 14)。
        # 后续 target image、adv image 以及不同 CLIP 的 local token 都必须从这同一个 mask 派生，
        # 这样才能做到真正的 symmetric/shared masking。
        self.current_mask_indices = []
        self.current_mask_grid_shape = None
        self.lambda_sem = 1.2
        self.lambda_rel = lambda_rel
        # 新增的 attention alignment 先给一个较小权重：
        # - 语义 OT 和 Gram relation 仍然是主项；
        # - attention relation 作为辅助约束，避免一开始就因为权重过大把优化拉偏。
        self.lambda_attn = lambda_attn
        # 当 attention 权重被显式设为 0 时，视作关闭这条辅助分支。
        # 这样 cluster=5 就可以只保留语义 + Gram relation，而不参与 attention alignment。
        self.use_attention_alignment = self.lambda_attn > 0

        # 温度退火参数 (Temperature Annealing)
        self.step_count = 0      # 记录当前调用的步数
        self.T_init = 2.0        # 初始温度（降低以加快早期聚焦）
        self.T_min = 0.5         # 最低温度（利用期）
        self.decay_rate = 0.95   # 温度衰减率

    # ================= Mask 辅助函数 =================
    def _mask_grid_shape(self, x: Tensor, patch_size: int = 16):
        """根据输入图像尺寸构造公共 mask 网格。

        这里的公共网格用于“选区域”，不等同于每个 CLIP backbone 的真实 token 网格。
        例如当前输入分辨率为 224，patch_size=16 时，公共 mask 网格为 14x14；
        之后会通过 project_mask_indices_to_tokens() 投影到 B16/B32/Laion 各自的 token 网格。
        """
        _, _, height, width = x.shape
        return height // patch_size, width // patch_size

    def sample_patch_mask_indices(self, x: Tensor, patch_size: int = 16):
        if random.random() > self.erasing_prob:
            return [], self._mask_grid_shape(x, patch_size=patch_size)

        _, _, height, width = x.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask = int(total_patches * random.uniform(self.erasing_scale[0], self.erasing_scale[1]))
        if num_mask == 0:
            return [], (num_patches_h, num_patches_w)

        return random.sample(range(total_patches), num_mask), (num_patches_h, num_patches_w)

    def select_saliency_mask_indices(self, x: Tensor, patch_size: int = 16):
        if random.random() > self.erasing_prob:
            return [], self._mask_grid_shape(x, patch_size=patch_size)

        avg_saliency, var_saliency, num_patches_h, num_patches_w = self.compute_saliency_scores(x)
        total_patches = num_patches_h * num_patches_w
        num_mask = int(total_patches * random.uniform(self.erasing_scale[0], self.erasing_scale[1]))
        if num_mask == 0:
            return [], (num_patches_h, num_patches_w)

        # 这里保留你要求的“mask 三个 CLIP 共同最关注区域”的设计：
        # - avg_saliency 高：说明多个 CLIP 普遍关注该区域；
        # - var_saliency 低：说明不同 CLIP 对该区域的关注更一致；
        # 因此用 mean - 0.5 * variance 作为共识关注度，并严格取 top-k，而不是随机采样。
        base_mask_weight = avg_saliency
        consensus_mask_weight = base_mask_weight - 0.5 * var_saliency
        consensus_mask_weight = torch.clamp(consensus_mask_weight, min=0.0)
        num_mask = min(num_mask, total_patches)
        _, topk_indices = torch.topk(consensus_mask_weight, k=num_mask, largest=True, sorted=False)
        return topk_indices.cpu().tolist(), (num_patches_h, num_patches_w)

    def apply_indices_mask_to_image(
        self,
        x: Tensor,
        masked_patch_indices: list = None,
        mask_grid_shape: tuple = None,
    ):
        """把共享 top-k mask 真实应用到图像像素上。

        这一步对应 Locality Alignment / MaskEmbed 里的 masked image query：
        teacher/encoder 看到的是 m(x)，而不是完整图像。为了避免黑块 artifact，
        被 mask 的 patch 不填 0，而是填 CLIP 的 dataset mean（注意当前数据管线在
        normalizer 前是 0~255，所以这里使用 mean * 255）。

        Args:
            x: 图像张量，形状通常为 [B, 3, H, W]，数值范围为 0~255。
            masked_patch_indices: 公共 mask 网格上的 flat indices。
            mask_grid_shape: 公共 mask 网格尺寸，例如 (14, 14)。
        """
        if masked_patch_indices is None:
            masked_patch_indices = self.current_mask_indices
        if mask_grid_shape is None:
            mask_grid_shape = self.current_mask_grid_shape
        if not masked_patch_indices or mask_grid_shape is None:
            return x

        x_masked = x.clone()
        _, channels, height, width = x.shape
        num_patches_h, num_patches_w = mask_grid_shape

        # CLIP mean in RGB order. 当前输入在 normalizer 前是 0~255，因此乘 255。
        # 如果后续数据管线改成 0~1，这里需要同步改成不乘 255。
        fill_value = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 3, 1, 1) * 255.0
        if channels != 3:
            fill_value = torch.zeros((1, channels, 1, 1), dtype=x.dtype, device=x.device)

        for patch_idx in masked_patch_indices:
            patch_row = patch_idx // num_patches_w
            patch_col = patch_idx % num_patches_w

            # 用比例切分而不是固定 patch_size，确保 crop 后只要分辨率一致/接近，
            # 同一个公共 grid index 仍然对应同一相对空间位置。
            start_h = int(round(patch_row * height / num_patches_h))
            end_h = int(round((patch_row + 1) * height / num_patches_h))
            start_w = int(round(patch_col * width / num_patches_w))
            end_w = int(round((patch_col + 1) * width / num_patches_w))
            x_masked[:, :, start_h:end_h, start_w:end_w] = fill_value

        return x_masked

    def apply_current_mask_to_image(self, x: Tensor):
        """给 attack loop 使用：把当前 target 选出的共享 mask 应用到 adv image 上。"""
        return self.apply_indices_mask_to_image(
            x,
            masked_patch_indices=self.current_mask_indices,
            mask_grid_shape=self.current_mask_grid_shape,
        )

    def apply_patch_mask_to_image(self, x: Tensor, patch_size: int = 16):
        x_masked = x.clone()
        _, _, height, width = x.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        masked_patch_indices, mask_grid_shape = self.sample_patch_mask_indices(x, patch_size=patch_size)
        if not masked_patch_indices:
            return x, []
        x_masked = self.apply_indices_mask_to_image(x, masked_patch_indices, mask_grid_shape)
        return x_masked, masked_patch_indices

    def apply_token_mask_to_embedding(self, embedding: Tensor, masked_patch_indices: list):
        # 兼容旧逻辑的 token zeroing 函数；当前主路径已经改为 visible-only alignment，
        # 不再直接把 token 置零，而是用 get_visible_token_indices() 只取未被 mask 的 tokens。
        if not masked_patch_indices:
            return embedding
        embedding_masked = embedding.clone()
        if embedding.dim() == 3:
            batch_size, num_tokens, feature_dim = embedding.shape
        else:
            num_tokens, feature_dim = embedding.shape
            batch_size = 1
            embedding_masked = embedding_masked.unsqueeze(0)
        for patch_idx in masked_patch_indices:
            token_idx = patch_idx
            if token_idx < num_tokens:
                embedding_masked[:, token_idx, :] = 0
        if embedding.dim() == 2:
            embedding_masked = embedding_masked.squeeze(0)
        return embedding_masked

    def project_mask_indices_to_tokens(
        self,
        masked_patch_indices: list,
        source_grid_shape: tuple,
        num_tokens: int,
        device: torch.device,
    ):
        """把公共 top-k mask 投影到当前 CLIP 的 local token 网格。

        为什么需要这一步：
        - B16 的 local token 网格是 14x14；
        - B32 的 local token 网格是 7x7；
        - Laion G-14 的 local token 网格是 16x16。
        如果直接把 14x14 的 flat index 用到所有模型上，B32/Laion 的空间位置会错位。
        因此先把公共 mask map resize 到当前模型的 token grid，再取对应 token index。
        """
        if not masked_patch_indices or source_grid_shape is None:
            return torch.empty(0, dtype=torch.long)

        token_side = int(math.sqrt(num_tokens))
        if token_side * token_side != num_tokens:
            # 理论上 CLIP ViT 的 patch token 都是正方形网格；如果遇到非正方形，
            # 保守退化为“只保留合法 index”，避免 shape 推断错误导致崩溃。
            valid = [idx for idx in masked_patch_indices if idx < num_tokens]
            return torch.tensor(valid, dtype=torch.long)

        source_h, source_w = source_grid_shape
        # 这个 mask 投影只处理几十/几百个 bool 值，没必要占用 CUDA kernel。
        # 放在 CPU 上做可以减少 attack loop 中的小 kernel 同步点；如果前面某个大模型
        # kernel 已经异步失败，也能避免错误被误报到这里的 torch.nonzero。
        mask_map = torch.zeros((1, 1, source_h, source_w), dtype=torch.float32)
        for patch_idx in masked_patch_indices:
            row = patch_idx // source_w
            col = patch_idx % source_w
            if 0 <= row < source_h and 0 <= col < source_w:
                mask_map[:, :, row, col] = 1.0

        token_mask = F.interpolate(mask_map, size=(token_side, token_side), mode="nearest")
        token_mask = token_mask.view(-1) > 0.5
        return torch.nonzero(token_mask, as_tuple=False).flatten().long()

    def get_visible_token_indices(self, embedding: Tensor):
        """返回当前模型中未被共享 mask 覆盖的 token indices。

        semantic OT 和 relation Gram 都只在 visible tokens 上计算，避免 masked token
        的均值 patch / 零向量参与 Sinkhorn transport 或 Gram 矩阵，导致对齐目标被稀释。
        """
        num_tokens = embedding.shape[-2]
        device = embedding.device
        if not self.use_random_erasing or not self.current_mask_indices:
            return torch.arange(num_tokens, dtype=torch.long, device=device)

        masked_token_indices = self.project_mask_indices_to_tokens(
            self.current_mask_indices,
            self.current_mask_grid_shape,
            num_tokens,
            device,
        )
        token_mask = torch.zeros(num_tokens, dtype=torch.bool)
        if masked_token_indices.numel() > 0:
            token_mask[masked_token_indices] = True
        visible_indices = torch.nonzero(~token_mask, as_tuple=False).flatten().long()

        # 极端情况下 top-k 投影后可能覆盖了较小 token grid 的全部位置。
        # 为了避免空 tensor 让 OT/Sinkhorn 崩掉，这里退回到全 token 对齐。
        if visible_indices.numel() == 0:
            visible_indices = torch.arange(num_tokens, dtype=torch.long)
        return visible_indices.to(device=device, non_blocking=True)
    
    @torch.no_grad()
    def compute_saliency_scores(self, x: Tensor):
        all_saliency_scores = []
        batch, channels, height, width = x.shape
        patch_size = 16
        target_h, target_w = height // patch_size, width // patch_size 
        for model in self.extractors:
            inputs = model.normalizer(x.to(x.device))
            outputs = model.model.vision_model(pixel_values=inputs, output_attentions=True)
            patch_importance = self._compute_attention_based_importance(outputs.attentions) 
            current_num_patches = patch_importance.shape[0]
            current_side = int(np.sqrt(current_num_patches))
            importance_2d = patch_importance.view(1, 1, current_side, current_side)
            rescaled_importance = F.interpolate(importance_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
            all_saliency_scores.append(rescaled_importance.view(-1))
        stacked_scores = torch.stack(all_saliency_scores)
        avg_saliency = stacked_scores.mean(dim=0)
        if len(self.extractors) > 1:
            var_saliency = stacked_scores.var(dim=0, unbiased=False)
            if var_saliency.max() > var_saliency.min():
                var_saliency = (var_saliency - var_saliency.min()) / (var_saliency.max() - var_saliency.min())
        else:
            var_saliency = torch.zeros_like(avg_saliency)
        return avg_saliency, var_saliency, target_h, target_w

    def _compute_attention_based_importance(self, attentions):
        last_attn = attentions[-1]  
        cls_to_patches = last_attn[:, :, 0, 1:]  
        avg_attention_received = last_attn[:, :, :, 1:].mean(dim=2)  
        combined_attention = (cls_to_patches + avg_attention_received) / 2.0  
        low_attention_threshold = combined_attention.median(dim=2, keepdim=True)[0]  
        is_low_attention = combined_attention < low_attention_threshold  
        low_attention_ratio = is_low_attention.float().mean(dim=1).squeeze(0)  
        importance_scores = 1.0 - low_attention_ratio  
        avg_attention_magnitude = combined_attention.mean(dim=1).squeeze(0)  
        if avg_attention_magnitude.max() > avg_attention_magnitude.min():
            normalized_magnitude = (avg_attention_magnitude - avg_attention_magnitude.min()) / \
                                 (avg_attention_magnitude.max() - avg_attention_magnitude.min())
        else:
            normalized_magnitude = torch.ones_like(avg_attention_magnitude)
        final_importance = 0.7 * importance_scores + 0.3 * normalized_magnitude
        return final_importance

    

    def apply_saliency_mask_to_image(self, x: Tensor, patch_size: int = 16):
        masked_patch_indices, mask_grid_shape = self.select_saliency_mask_indices(x, patch_size=patch_size)
        if not masked_patch_indices:
            return x, []
        x_masked = self.apply_indices_mask_to_image(x, masked_patch_indices, mask_grid_shape)
        return x_masked, masked_patch_indices

   
    def reset_attack_state(self):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_attention.clear()
        self.previous_loss_list.clear()
        self.previous_loss_local_list.clear()
        self.current_mask_indices.clear()
        self.current_mask_grid_shape = None
        self.step_count = 0

    # 🔥 核心修正 1：计算关系矩阵的函数
    def compute_relation_matrix(self, embedding: Tensor):
        """计算特征的 Gram 关系矩阵 R = E E^T / sqrt(d)"""
        d = embedding.shape[-1]
        # embedding 形状通常是 (num_tokens, feature_dim)
        R = torch.matmul(embedding, embedding.transpose(-1, -2)) / math.sqrt(d)
        return R

    def compute_visible_attention_matrix(self, attention: Tensor, visible_indices: Tensor):
        """
        从完整 local-local attention 中裁出当前 shared mask 下的 visible 子矩阵。

        为什么这里还要做一次行归一化：
        - 原始 self-attention 在完整 token 集上做过 softmax，每一行和约等于 1；
        - 但我们把 masked token 裁掉之后，保留下来的 visible 子矩阵每一行和会小于 1；
        - 若不重新归一化，不同 mask ratio 下的数值尺度会漂，attention loss 会混入“可见 token 数量”
          这个额外因素，而不是纯粹比较关系结构本身。
        """
        visible_attention = attention.index_select(0, visible_indices).index_select(1, visible_indices)
        visible_attention = torch.clamp(visible_attention, min=0.0)
        visible_attention = visible_attention / (visible_attention.sum(dim=-1, keepdim=True) + 1e-8)
        return visible_attention

    # ========================================================

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        self.ground_truth_attention.clear()
        self.current_mask_indices.clear() # 清除上一轮的记录

        masked_patch_indices = []
        mask_grid_shape = self._mask_grid_shape(x, patch_size=16)
        
        if self.use_random_erasing:
            if self.use_saliency_mask:
                masked_patch_indices, mask_grid_shape = self.select_saliency_mask_indices(x, patch_size=16)
            else:
                masked_patch_indices, mask_grid_shape = self.sample_patch_mask_indices(x, patch_size=16)

        # 🔥 核心修正 2：
        # 用 target image 选出本 step 的共享 top-k mask，并保存下来给 adv image 复用。
        # 同时 target 自己也先做 image-level mask，再提取 f(m(x_t))，对应 Locality Alignment
        # 里“用 masked image query teacher”的思路。
        self.current_mask_indices = masked_patch_indices
        self.current_mask_grid_shape = mask_grid_shape
        x_masked = self.apply_indices_mask_to_image(x, masked_patch_indices, mask_grid_shape)

        for model in self.extractors:
            if self.use_attention_alignment:
                # target 端除了 global/local feature，还额外缓存 masked view 下的 local attention。
                # 后面 adv 端会在同一个 shared visible token 集上去模仿它。
                x_tensor, x_embedding, x_attention = model.global_local_attention_features(x_masked.to(x.device))
            else:
                x_tensor, x_embedding = model.global_local_features(x_masked.to(x.device))
                x_attention = None
            x_embedding = x_embedding.squeeze(0)
            x_embedding = F.normalize(x_embedding, dim=-1)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(x_embedding.unsqueeze(0))
            self.ground_truth_attention.append(None if x_attention is None else x_attention.squeeze(0))

    def __call__(self, features: List[Tensor], features_local: List[Tensor], features_attention: List[Tensor] = None):
        total_losses = []
        
        for index in range(len(self.extractors)):
            gt = self.ground_truth[index]
            gt_local = self.ground_truth_local[index].squeeze(0)
            
            feature = features[index].unsqueeze(0)
            feature_local = features_local[index].squeeze(0)

            feature_local = F.normalize(feature_local, dim=-1)
            gt_local = F.normalize(gt_local, dim=-1)

            # 🔥 核心修正 3：
            # 不同 CLIP 的 token 网格不同，不能直接复用公共 flat index。
            # 这里先把共享 top-k mask 投影到当前模型 token 网格，再只取 visible tokens。
            # 这相当于在 token/embedding 层也使用同一个 shared mask，但避免 masked tokens
            # 参与 OT 和 relation loss。
            visible_indices = self.get_visible_token_indices(feature_local)
            feature_local_visible = feature_local.index_select(0, visible_indices)
            gt_local_visible = gt_local.index_select(0, visible_indices)

            # 1. 局部内容对齐 (Local Content Alignment via OT)
            feat_loss = OT(gt, feature)
            semantic_alignment = OT(gt_local_visible, feature_local_visible)
            
            # 🔥 核心修正 4：局部关系对齐 (Local Relational Alignment via MSE)
            R_gt = self.compute_relation_matrix(gt_local_visible)
            R_adv = self.compute_relation_matrix(feature_local_visible)
            relational_alignment = -F.mse_loss(R_adv, R_gt)

            # 新增：attention relation alignment。
            # 这里对齐的不是“token feature 本身”，而是“每个局部区域如何关注其他局部区域”的分布。
            # 它和上面的 Gram relation 是互补关系：
            # - Gram 更像静态相似性拓扑；
            # - Attention 更像 transformer 内部的动态依赖模式。
            attention_alignment = feature_local_visible.new_tensor(0.0)
            if self.use_attention_alignment and features_attention is not None:
                gt_attention = self.ground_truth_attention[index]
                feature_attention = features_attention[index]
                if gt_attention is not None and feature_attention is not None:
                    gt_attention = gt_attention.squeeze(0) if gt_attention.dim() == 3 else gt_attention
                    feature_attention = feature_attention.squeeze(0) if feature_attention.dim() == 3 else feature_attention

                    gt_attention_visible = self.compute_visible_attention_matrix(gt_attention, visible_indices)
                    feature_attention_visible = self.compute_visible_attention_matrix(feature_attention, visible_indices)

                    # 当前 attack 的整体目标是“最大化总 score”；
                    # 因此 attention 这类距离项要取负号，表示“距离越小，score 越大”。
                    attention_alignment = -F.mse_loss(feature_attention_visible, gt_attention_visible)

            total_loss_i = (
                feat_loss
                + self.lambda_sem * semantic_alignment
                + self.lambda_rel * relational_alignment
                + self.lambda_attn * attention_alignment
            )
            total_losses.append(total_loss_i)
        
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            raw_ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            ratio = max(0.2, min(5.0, raw_ratio))
            weights.append(ratio)

        T = max(self.T_min, self.T_init * (self.decay_rate ** self.step_count))
        self.step_count += 1
        
        K = len(weights)
        weights_np = np.array(weights)
        
        weights_np_shifted = weights_np - np.max(weights_np)
        weights_softmax = np.exp(weights_np_shifted / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K 

        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss
