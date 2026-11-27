import torch

from torchvision.ops import box_convert, box_iou


def Ap(cls, box, targets, APx, records):  # B 40 81 ,B 40 4,targets is list,len is B
    # records = {c: {"scores": [], "tp": [], "fp": [], "gt_num": 0} for c in range(80)}
    p = cls.softmax(-1)  # B 40 81

    (
        score,
        label,
    ) = p[
        :, :, :-1
    ].max(  # No background
        -1
    )  # B 40
    for i in range(len(targets)):
        classes = torch.cat([targets[i]["cls"], label[i]]).unique()

        for onecls in classes:
            pred_mask = label[i] == onecls
            gt_mask = targets[i]["cls"] == onecls

            pred_boxes = box[i][pred_mask]
            pred_scores = score[i][pred_mask]
            gt_boxes = targets[i]["gtbox"][gt_mask]

            indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[indices]
            pred_scores = pred_scores[indices]  # sorted

            pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
            gt_xyxy = box_convert(gt_boxes, "cxcywh", "xyxy")

            iou = box_iou(pred_xyxy, gt_xyxy)
            # like assume pre have 4,gt have 3,  4*3.(have sorted)
            boolvisited = torch.zeros(
                len(gt_boxes), dtype=torch.bool, device=iou.device
            )

            tp = torch.zeros(len(pred_boxes), device=iou.device)
            fp = torch.zeros(len(pred_boxes), device=iou.device)

            for j in range(len(pred_boxes)):
                current_iou = iou[j]  # [M]
                current_iou[boolvisited] = -1  # visited set is -1
                maxiou, index = current_iou.max(-1)  # get the remain gtbox
                if maxiou < APx:
                    fp[j] = 1
                    continue
                boolvisited[index] = True
                tp[j] = 1

            c = onecls.item()
            records[c]["scores"].append(pred_scores)
            records[c]["tp"].append(tp)
            records[c]["fp"].append(fp)
            records[c]["gt_num"] += len(gt_boxes)


def compute_ap(records):
    aps = []
    for record in records.values():
        # 1. 获取该类别 GT 总数
        thisgtnum = record["gt_num"]
        if thisgtnum == 0:
            continue
        if len(record["tp"]) == 0:
            aps.append(0.0)
            continue
        # 2. 合并 scores、tp、fp
        scores = torch.cat(record["scores"], -1)
        tp = torch.cat(record["tp"], -1)
        fp = torch.cat(record["fp"], -1)
        # 3. 全局同步排序
        indic = torch.argsort(scores, descending=True)
        scores = scores[indic]
        tp = tp[indic]
        fp = fp[indic]
        # 4. 累积求和
        precison = torch.zeros([len(scores)])
        recall = torch.zeros([len(scores)])

        cum_tp = tp.cumsum(0)
        cum_fp = fp.cumsum(0)
        precison = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / thisgtnum
        # 插值法
        for i in range(len(precison) - 2, -1, -1):
            if precison[i + 1] > precison[i]:
                precison[i] = precison[i + 1]

        AP = recall[0] * precison[0]

        for i in range(1, len(precison)):
            AP += (recall[i] - recall[i - 1]) * precison[i]
        aps.append(AP)
    return sum(aps) / len(aps) if len(aps) > 0 else float("nan")


if __name__ == "__main__":

    # [1, 3, 81]
    cls = torch.zeros(1, 3, 81)

    cls[0, 0, 0] = 10
    cls[0, 1, 0] = 9
    cls[0, 2, 0] = 8

    # [1, 3, 4] cxcywh
    box = torch.tensor(
        [
            [
                [0.25, 0.25, 0.2, 0.2],  # A
                [0.26, 0.25, 0.2, 0.2],  # B
                [0.75, 0.75, 0.2, 0.2],  # C
            ]
        ]
    )

    targets = [
        {
            "cls": torch.tensor([0, 0]),
            "gtbox": torch.tensor(
                [[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]  # GT0  # GT1
            ),
        }
    ]

    records = {c: {"scores": [], "tp": [], "fp": [], "gt_num": 0} for c in range(80)}
    Ap(cls, box, targets, 0.5, records)

    print(records[0])
    print(compute_ap(records))
