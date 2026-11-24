import os

import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.ops import box_convert


class COCODataset(CocoDetection):
    def __init__(self, data_root, split, image_size=640):
        # split："train2017" 或 "val2017"。
        super().__init__(
            root=os.path.join(data_root, split),
            annFile=os.path.join(data_root, "annotations", f"instances_{split}.json"),
        )

        # COCO 的 category_id 不连续，统一映射成 0～79。
        # 第 80 类留给 no-object，train/val 使用同一套排序规则。
        category_ids = sorted(self.coco.getCatIds())
        self.class_to_idx = {
            category_id: i for i, category_id in enumerate(category_ids)
        }
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        W, H = image.size

        # crowd 是群体标注，不作为单个物体参与匹配。
        annotations = [obj for obj in annotations if obj["iscrowd"] == 0]
        boxes = torch.tensor(
            [obj["bbox"] for obj in annotations], dtype=torch.float32
        ).reshape(-1, 4)  # [N,4]，COCO 原始格式是左上角 xywh。
        labels = torch.tensor(
            [self.class_to_idx[obj["category_id"]] for obj in annotations],
            dtype=torch.long,
        )  # [N]

        # 转 xyxy，裁到图片内，再保留宽高为正的标注框。
        boxes = box_convert(boxes, "xywh", "xyxy")
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        labels = labels[keep]

        # DETR 使用归一化 cxcywh：[N,4]。
        # 图片直接缩放到正方形，按原图宽高归一化后不需要再改框。
        boxes = box_convert(boxes, "xyxy", "cxcywh")
        boxes = boxes / torch.tensor([W, H, W, H], dtype=torch.float32)
        image = self.transform(image)  # [3,640,640]
        target = {"labels": labels, "boxes": boxes}
        return image, target


def collate_fn(batch):
    # 图片大小已统一，直接 stack；每张图目标数不同，target 保留为 list。
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
