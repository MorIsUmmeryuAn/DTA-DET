import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner



class FocalLoss(nn.Module):
    """Focal Loss for classification"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class SHaSMLoss(nn.Module):
    """空间感知难样本挖掘损失函数"""

    def __init__(self, num_classes=80, alpha=0.5, beta=1.0, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 前景权重
        self.beta = beta  # 难样本权重
        self.gamma = gamma  # 聚焦参数

        # 基础损失函数
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: 模型预测结果 [B, N, C+4]
            targets: 真实标签 [B, M, 5] (class_idx, x, y, w, h)
        Returns:
            加权后的损失值
        """
        # 1. 计算基础分类和回归损失
        cls_loss = self.bce_loss(preds[..., 4:], targets[..., 0].long())
        reg_loss = self.focal_loss(preds[..., :4], targets[..., 1:5])

        # 2. 识别难样本 (预测与GT不一致的样本)
        with torch.no_grad():
            # 计算预测置信度
            pred_conf = torch.sigmoid(preds[..., 4:]).max(dim=-1)[0]
            # 计算IoU
            iou = bbox_iou(preds[..., :4], targets[..., 1:5])
            # 难样本权重 = (1 - IoU) * (1 - confidence)
            hard_weight = (1 - iou) * (1 - pred_conf)

        # 3. 前景/背景区分
        fg_mask = targets[..., 0] >= 0  # 前景样本
        bg_mask = ~fg_mask  # 背景样本

        # 4. 应用权重
        # 前景样本获得更高权重
        cls_loss[fg_mask] *= self.alpha
        reg_loss[fg_mask] *= self.alpha

        # 难样本获得额外权重
        cls_loss *= (1 + self.beta * hard_weight)
        reg_loss *= (1 + self.beta * hard_weight)

        # 5. 聚焦困难样本
        cls_loss = cls_loss * ((1 - pred_conf) ** self.gamma)
        reg_loss = reg_loss * ((1 - iou) ** self.gamma)

        # 返回平均损失
        return cls_loss.mean() + reg_loss.mean()


class v8DetectionLossHard(nn.Module):
    """兼容YOLOv8原有损失函数的增强版本"""

    def __init__(self, model):
        super().__init__()
        # 原有YOLOv8损失函数组件
        self.stride = model.stride
        self.nc = model.model[-1].nc
        self.no = model.model[-1].no
        self.reg_max = model.model[-1].reg_max if hasattr(model.model[-1], 'reg_max') else 0

        # 原有损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=False).to(model.device)
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

        # 我们的SHaSM损失
        self.shasm_loss = SHaSMLoss(num_classes=self.nc)

        # 损失权重
        self.loss_weights = {
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5,
            'shasm': 1.0  # SHaSM损失权重
        }
        self.ctkd_weight = 0.5  # 可调整

    def forward(self, preds, batch):
        """计算并返回所有损失"""
        # 原始YOLOv8损失计算
        device = preds[0].device
        loss = torch.zeros(4, device=device)  # [class, iou, dfl, shasm]

        # 获取预测和分配目标
        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_distri, pred_scores = pred_distri.permute(0, 2, 1).contiguous(), pred_scores.permute(0, 2, 1).contiguous()

        # 任务对齐分配
        targets = self.assigner(pred_scores.detach().sigmoid(), (pred_distri.detach() * self.stride).view(-1, 4),
                                batch['cls'], batch['bboxes'], batch['img'].shape[-2:])

        # 分类损失
        target_scores = targets['target_scores']
        loss[0] = self.bce(pred_scores, target_scores).mean() * self.loss_weights['class']

        # 回归损失
        target_bboxes = targets['target_bboxes'] / self.stride
        loss[1], loss[2] = self.bbox_loss(pred_distri, target_bboxes) * self.loss_weights['iou']

        # SHaSM损失计算
        pred_boxes = torch.cat(preds, 1)[..., :4]  # 获取所有预测框
        gt_boxes = batch['bboxes']
        gt_classes = batch['cls']

        # 准备SHaSM输入 (简化版，实际实现需要更复杂的匹配)
        shasm_input = torch.cat([
            pred_boxes,  # 预测框坐标
            pred_scores.unsqueeze(-1)  # 预测分数
        ], dim=-1)

        shasm_target = torch.cat([
            gt_boxes,  # GT框坐标
            gt_classes.unsqueeze(-1).float()  # GT类别
        ], dim=-1)

        loss[3] = self.shasm_loss(shasm_input, shasm_target) * self.loss_weights['shasm']

        # 总损失
        total_loss = loss.sum()

        # 添加CTKD损失
        if hasattr(self, 'ctkd') and 'ctkd_loss' in preds:
            ctkd_loss = preds['ctkd_loss']
            total_loss += self.ctkd_weight * (ctkd_loss['struct_loss'] + ctkd_loss['contrast_loss'])

        return total_loss, torch.cat((loss, loss.sum().unsqueeze(0))).detach()
