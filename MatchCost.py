import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.cost_class = 1
        self.cost_bbox = 5
        self.cost_giou = 2

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):

        indices = []

        for b in range(pred_logits.shape[0]):
            logits = pred_logits[b]  # [100, 81]

            boxes = pred_boxes[b]  # [100, 4]

            gt_labels = targets[b]["labels"]  # like [1, 2, 3]
            gt_boxes = targets[b]["boxes"]

            # TODO 1: Classification Cost
            prob = logits.softmax(dim=-1)
            cost_class = -prob[:, gt_labels]
            # like [100, 3]，表示每个预测框对每个gt框的分类损失
            # TODO 2: L1 Cost
            cost_bbox = torch.cdist(boxes, gt_boxes, p=1)
            # cdist 是 pairwise distance（成对距离） 计算函数。
            # TODO 3: GIoU Cost
            # cxcywh -> xyxy
            pred_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            gt_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")

            cost_giou = -generalized_box_iou(pred_xyxy, gt_xyxy)
            # [100,N_gt]
            # [100, 3]
            # TODO 4: 组合 Cost 矩阵
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            # TODO 5: Hungarian Assignment
            pred_indices, gt_indices = linear_sum_assignment(C.cpu())
            indices.append((pred_indices, gt_indices))
            # like pred_idx = [8, 21, 65]
            # gt_idx   = [1,  2,  0]
        return indices  # 所有图片匹配结束后，统一返回。
