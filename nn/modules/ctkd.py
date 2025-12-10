# ultralytics/nn/modules/ctkd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class PrototypeBank(nn.Module):
    """类别原型记忆库"""

    def __init__(self, num_classes, feature_dim, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('counts', torch.zeros(num_classes))
        self.initialized = False

    def update(self, features, labels):
        """动量更新原型"""
        if not self.initialized:
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.any():
                    self.prototypes[c] = features[mask].mean(0)
                    self.counts[c] = mask.sum()
            self.initialized = True
        else:
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.any():
                    new_mean = features[mask].mean(0)
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * new_mean
                    self.counts[c] = self.momentum * self.counts[c] + (1 - self.momentum) * mask.sum()

    def get_prototypes(self):
        return self.prototypes


class CTKD(nn.Module):
    """类别感知拓扑知识扩散模块"""

    def __init__(self, num_classes, feature_dim=256, temperature=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature

        # 双原型库
        self.src_prototype = PrototypeBank(num_classes, feature_dim)
        self.tgt_prototype = PrototypeBank(num_classes, feature_dim)

        # 结构矩阵
        self.register_buffer('struct_matrix', torch.zeros(num_classes, num_classes))

    def compute_structure_matrix(self, prototypes):
        """计算原型间结构关系矩阵"""
        prototypes = F.normalize(prototypes, p=2, dim=1)
        sim_matrix = torch.mm(prototypes, prototypes.t())
        return sim_matrix

    def forward(self, x):
        """简化forward方法用于模型初始化"""
        if isinstance(x, (list, tuple)) and len(x) == 4:
            # 训练时完整参数
            return self._forward_train(*x)
        else:
            # 初始化时仅传递特征
            return x

    def _forward_train(self, src_feats, src_labels, tgt_feats=None, tgt_labels=None):
        """实际训练逻辑"""
        """
        Args:
            src_feats: [N_s, D] 源域特征
            src_labels: [N_s] 源域标签
            tgt_feats: [N_t, D] 目标域特征 (可选)
            tgt_labels: [N_t] 目标域伪标签 (可选)
        Returns:
            dict: 包含损失和原型信息
        """
        # 1. 更新原型库
        self.src_prototype.update(src_feats.detach(), src_labels)
        if tgt_feats is not None and tgt_labels is not None:
            self.tgt_prototype.update(tgt_feats.detach(), tgt_labels)

        # 2. 结构一致性损失
        src_protos = self.src_prototype.get_prototypes()
        src_struct = self.compute_structure_matrix(src_protos)

        if tgt_feats is not None and tgt_labels is not None:
            tgt_protos = self.tgt_prototype.get_prototypes()
            tgt_struct = self.compute_structure_matrix(tgt_protos)
            struct_loss = F.mse_loss(src_struct, tgt_struct)
        else:
            struct_loss = torch.tensor(0.0, device=src_feats.device)

        # 3. 类别感知对比学习
        src_protos_norm = F.normalize(src_protos, p=2, dim=1)  # [C, D]
        src_feats_norm = F.normalize(src_feats, p=2, dim=1)  # [N_s, D]

        logits = torch.mm(src_feats_norm, src_protos_norm.t()) / self.temperature
        contrast_loss = F.cross_entropy(logits, src_labels)

        # 更新结构矩阵
        self.struct_matrix = src_struct.detach()

        return {
            'struct_loss': struct_loss,
            'contrast_loss': contrast_loss,
            'src_prototypes': src_protos,
            'struct_matrix': src_struct
        }