import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou


def Loss(pred_logits, pred_boxes, targets, indices):
    B, N, C = pred_logits.shape

    # 1. 分类损失准备
    weight = torch.ones(C, device=pred_logits.device)
    weight[C - 1] = 0.1  # no-object

    target_classes = torch.full(
        (B, N), C - 1, dtype=torch.long, device=pred_logits.device
    )

    # 2. 初始化边界框损失
    total_l1 = pred_boxes.new_zeros(())
    total_giou = pred_boxes.new_zeros(())
    num_boxes = 0

    for b, (pred_idx, gt_idx) in enumerate(indices):

        # Hungarian 匹配对应的类别标签
        target_classes[b, pred_idx] = targets[b]["labels"][gt_idx]

        # 取出匹配成功的预测框、GT框
        matched_pred = pred_boxes[b, pred_idx]
        matched_gt = targets[b]["boxes"][gt_idx]

        # L1 Loss：直接使用 cxcywh
        total_l1 += F.l1_loss(matched_pred, matched_gt, reduction="sum")

        # GIoU Loss：需要转换成 xyxy
        pred_xyxy = box_convert(matched_pred, "cxcywh", "xyxy")
        gt_xyxy = box_convert(matched_gt, "cxcywh", "xyxy")

        giou = generalized_box_iou(pred_xyxy, gt_xyxy)

        # 只取匹配成功的对角线
        total_giou += (1 - torch.diag(giou)).sum()

        num_boxes += len(gt_idx)

    # 3. 所有图片的类别填好后，再统一计算 Classification Loss。
    loss_cls = F.cross_entropy(
        pred_logits.transpose(1, 2), target_classes, weight=weight
    )

    # 4. Bounding Box Loss
    # 无目标图片仍计算 no-object 分类；无 GT 时，框损失保持为 0。
    num_boxes = max(num_boxes, 1)
    loss_bbox = total_l1 / num_boxes

    # 5. GIoU Loss
    loss_giou = total_giou / num_boxes

    # 6. DETR 总损失
    total_loss = loss_cls + 5 * loss_bbox + 2 * loss_giou

    return total_loss
