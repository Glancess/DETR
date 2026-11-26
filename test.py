import os

import torchvision
from moxie import model, validation, draw, Getmatcher
from PIL import Image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch
from moxie import MyDataset
from torch.utils.data import random_split


def Getrealimage():
    dataset = MyDataset("/opt/data/private/datasets/yangzhuohan/coco128")

    val_num = round(len(dataset) * 0.2)
    train_num = len(dataset) - val_num

    train_dataset, val_dataset = random_split(
        dataset, [train_num, val_num], generator=torch.Generator().manual_seed(42)
    )
    show = []
    for index in train_dataset.indices[:10]:
        show.append(dataset.files[index])
    return show


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    mode = model()
    state = torch.load(
        "/home/yangzhuohan/project/DETR/overfitDetr.pt", map_location="cpu"
    )
    mode.load_state_dict(state)
    show = Getrealimage()
    for imgname in show:
        path = os.path.join(
            "/opt/data/private/datasets/yangzhuohan/coco128/images/train2017",
            imgname,
        )
        img = Image.open(path).convert("RGB")
        W, H = img.size

        # -------------
        targets = []
        target = {}
        cls = []
        boxs = []
        labelpath = os.path.join(
            "/opt/data/private/datasets/yangzhuohan/coco128/labels/train2017",
            os.path.splitext(imgname)[0] + ".txt",
        )
        with open(
            labelpath,
            "r",
        ) as f:
            for line in f:
                parts = line.strip().split()
                cls.append(int(parts[0]))
                boxs.append([float(v) for v in parts[1:]])
            # 5. 分离类别和 bbox
            # 6. 返回 image, target
            cls = torch.tensor(cls, dtype=torch.long)  # [G]
            boxs = torch.tensor(boxs, dtype=torch.float32).view(-1, 4)  # [G,4]
            target["gtbox"] = boxs
            target["cls"] = cls
        targetboxxyxy = box_convert(target["gtbox"], in_fmt="cxcywh", out_fmt="xyxy")
        scale = targetboxxyxy.new_tensor([W, H, W, H])
        targetboxxyxy = targetboxxyxy * scale
        target["gtbox"] = targetboxxyxy
        targets.append(target)
        # ----------------------

        mode.eval()
        with torch.no_grad():
            boxlast, label, conf = validation(mode, img, 0.8)
            save_path = os.path.join("results", imgname)
            draw(img, boxlast, label, conf, targets=targets, save_path=save_path)
