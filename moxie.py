import math
import os


import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.optim.optimizer


def Getposition(H, W):
    position = torch.zeros([1, H * W, 256])
    for i in range(H * W):
        y = i // W
        x = i % W
        pe_y = torch.zeros(128)
        pe_x = torch.zeros(128)
        for j in range(64):
            div = 10000 ** (2 * j / 128)

            # y 坐标
            pe_y[2 * j] = math.sin(y / div)
            pe_y[2 * j + 1] = math.cos(y / div)

            # x 坐标
            pe_x[2 * j] = math.sin(x / div)
            pe_x[2 * j + 1] = math.cos(x / div)
        position[:, i] = torch.cat([pe_x, pe_y], dim=0)
    return position


class MyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.WQ = nn.Linear(256, 256)
        self.WK = nn.Linear(256, 256)
        self.WV = nn.Linear(256, 256)
        self.WO = nn.Linear(256, 256)

    def forward(self, Q, K, V):
        """
        需要 位置+初始X
        """
        B, N, C = Q.shape
        B1, N1, C1 = K.shape
        # 在crossattention时候，D+E @ Q ，100 256，M+P）@K=400 256，故我们reshape的时候需要分开
        Q = self.WQ(Q)  # 1 400 256
        K = self.WK(K)  # 1 400 256
        V = self.WV(V)  # 1 400 256
        Q = Q.reshape(B, N, 8, 32).permute(0, 2, 1, 3)
        K = K.reshape(B, N1, 8, 32).permute(0, 2, 1, 3)
        V = V.reshape(B, N1, 8, 32).permute(0, 2, 1, 3)
        # like 1 8 400 32 ,8 heads,every head have 400token which have 32dim
        score = Q @ K.transpose(-2, -1) / (32**0.5)  # 1 8 400 400
        score = torch.softmax(score, -1)
        attention = score @ V  # [1, 8, 400, 32]
        # merge
        attention = attention.transpose(1, 2).reshape(B, N, -1)
        attention = self.WO(attention)
        return attention


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MyAttention()
        self.dropout1 = nn.Dropout(0)
        self.layernorm1 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 256))
        self.dropout2 = nn.Dropout(0)
        self.layernorm2 = nn.LayerNorm(256)

    def forward(self, X, P):
        attention = self.attn(X + P, X + P, X)
        result = self.dropout1(attention)
        result0 = self.layernorm1(X + result)
        result = self.ffn(result0)
        result = self.dropout2(result)
        result = self.layernorm2(result0 + result)
        return result


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = EncoderLayer()
        self.layer2 = EncoderLayer()
        self.layer3 = EncoderLayer()
        self.layer4 = EncoderLayer()
        self.layer5 = EncoderLayer()
        self.layer6 = EncoderLayer()

    def forward(self, X, P):
        r1 = self.layer1(X, P)
        r2 = self.layer2(r1, P)
        r3 = self.layer3(r2, P)
        r4 = self.layer4(r3, P)
        r5 = self.layer5(r4, P)
        r6 = self.layer6(r5, P)
        return r6


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("position", Getposition(20, 20), persistent=False)
        res = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(res.children())[:-2])
        self.conv1 = nn.Conv2d(2048, 256, 1)
        self.Encoder = Encoder()
        self.position = Getposition(20, 20)
        self.Decoder = Decoder()
        self.Eobj = nn.Parameter(torch.randn(40, 256))
        self.clshead = nn.Linear(256, 81)
        self.boxhead = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, X):

        X = self.backbone(X)
        X = self.conv1(X)
        X = X.flatten(2, 3).transpose(1, 2)  # transpose(两个位置)，permute=任意位置
        M = self.Encoder(X, self.position)
        D0 = X.new_zeros(X.size(0), self.Eobj.size(0), X.size(-1))
        result = self.Decoder(D0, self.Eobj, M, self.position)
        cls = self.clshead(result)
        box = self.boxhead(result)
        return cls, box


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.selfattn = MyAttention()
        self.crossattn = MyAttention()
        self.dropout1 = nn.Dropout(0)
        self.layernorm1 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0)
        self.layernorm2 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 256))
        self.layernorm3 = nn.LayerNorm(256)

    def forward(self, D, Eobj, M, P):
        slfattn = self.selfattn(D + Eobj, D + Eobj, D)
        slfattn = self.dropout1(slfattn)
        slfattn = self.layernorm1(slfattn + D)
        crosattn = self.crossattn(slfattn + Eobj, M + P, M)
        crosattn = self.dropout2(crosattn)
        result0 = self.layernorm2(crosattn + slfattn)
        result = self.ffn(result0)
        result = self.layernorm3(result + result0)
        return result


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DecoderLayer()
        self.layer2 = DecoderLayer()
        self.layer3 = DecoderLayer()
        self.layer4 = DecoderLayer()
        self.layer5 = DecoderLayer()
        self.layer6 = DecoderLayer()

    def forward(self, D, Eobj, M, P):
        r1 = self.layer1(D, Eobj, M, P)
        r2 = self.layer2(r1, Eobj, M, P)
        r3 = self.layer3(r2, Eobj, M, P)
        r4 = self.layer4(r3, Eobj, M, P)
        r5 = self.layer5(r4, Eobj, M, P)
        r6 = self.layer6(r5, Eobj, M, P)
        return r6


@torch.no_grad()
def Getmatcher(cls, box, targets):
    # cls =[B,100,81],box=[B,100,4],target is a dict,like { cls:[],gtbox:[[][][]]}

    indices = []
    for i in range(cls.shape[0]):
        prob = cls[i].softmax(-1)
        p = -prob[:, targets[i]["cls"]]  # like 100 3
        cost_bbox = torch.cdist(box[i], targets[i]["gtbox"], p=1)  # like 100 * 3
        prexyxy = box_convert(box[i], in_fmt="cxcywh", out_fmt="xyxy")
        target = box_convert(targets[i]["gtbox"], in_fmt="cxcywh", out_fmt="xyxy")
        giou = -(generalized_box_iou(prexyxy, target))  # 100 * 3
        # 假设GT有3个，3个类别。我们得到了100个预测框与3个类别的负的概率，拿到了pre预测到的box与GTbox的差距，拿到了pre与GTbox的GIOU。
        cost = p + 5 * cost_bbox + 2 * giou
        pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append((pred_idx, gt_idx))
    return indices


def Loss(indices, targets, cls, box):
    # 我们的 cls ，box 都是 B 100 81 / 4
    # 我们已经得到了  选出来的pre，以及对应的GT，我们要cls的loss。
    B, N, C = box.shape
    rightcls = torch.full((B, N), 80, dtype=torch.long, device=cls.device)
    weight = torch.ones(81, device=cls.device)
    weight[80] = 0.1
    gtboxnumber = 0
    boxallloss = 0
    giousloss = 0
    for i in range(B):
        pred_idx, gt_idx = indices[i]
        rightcls[i, pred_idx] = targets[i]["cls"][gt_idx]
        gtboxnumber += len(gt_idx)
        boxloss = F.l1_loss(
            box[i, pred_idx], targets[i]["gtbox"][gt_idx], reduction="sum"
        )
        boxallloss += boxloss
        prexyxy = box_convert(box[i, pred_idx], in_fmt="cxcywh", out_fmt="xyxy")
        targetxyxy = box_convert(
            targets[i]["gtbox"][gt_idx], in_fmt="cxcywh", out_fmt="xyxy"
        )
        giou = generalized_box_iou(prexyxy, targetxyxy)
        giousloss += (1 - torch.diag(giou)).sum()
    # like [B,100],but cls like [B,100,81]
    precls = cls.transpose(1, 2)  # the grammar command that cls must in first
    cls_loss = F.cross_entropy(precls, rightcls, weight=weight)
    with torch.no_grad():
        fg_loss, bg_loss, fg_acc, back_acc = classification_metrics(cls, rightcls)
        print(
            f"fg_loss:{fg_loss}, bg_loss:{bg_loss}, fg_acc:{fg_acc}, back_acc:{back_acc}"
        )
    boxallloss = boxallloss / gtboxnumber
    giousloss /= gtboxnumber
    print("_________________________")
    print("cls:", cls_loss.item())
    print("bbox:", boxallloss.item())
    print("giou:", giousloss.item())
    print("_________________________")
    return 2 * giousloss + 5 * boxallloss + cls_loss


def classification_metrics(cls, rightcls):  # B 40 81,B 40
    query_loss = F.cross_entropy(
        cls.transpose(1, 2), rightcls, reduction="none"
    )  # get shape B 40(i changed the query numbers)
    fore = rightcls != 80
    fg_loss = query_loss[fore].mean()
    bg_loss = query_loss[~fore].mean()
    last_precls = cls.argmax(-1)  # B 40,get the query pred cls
    booll = last_precls == rightcls  # get the correct match
    # 我们得到了 pre与right匹配的bool，也有 fore
    fore_match = (fore & booll).sum()
    fore_total = fore.sum()

    fg_acc = fore_match / fore_total

    back_acc = (~fore & booll).sum() / (~fore).sum()
    return fg_loss, bg_loss, fg_acc, back_acc


def train_one_epoch(optimizer, model, image, targets):

    cls, box = model(image)
    indices = Getmatcher(cls, box, targets)
    loss = Loss(indices, targets, cls, box)
    print(f"loss:{loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def collate_fn(batch):
    targets = []
    images = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images)
    return images, targets


from PIL import Image


def validation(model, X, threshold):
    model.eval()
    with torch.no_grad():
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((640, 640)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        W, H = X.size
        X = transform(X)
        X = X.unsqueeze(0)
        cls, box = model(X)  # like 1 50 81,1 50 4
        p = torch.softmax(cls[0], -1)
        print(p)
        conf, label = p.max(-1)
        print(conf)
        print(label)
        keep = (conf > threshold) & (label != 80)  # 背景类 80 排除掉    threshold = 0.4
        box = box[0][keep]  # [N, 4] 归一化 cxcywh
        label = label[keep]
        conf = conf[keep]
        boxxyxy = box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
        boxlast = []
        scale = boxxyxy.new_tensor([W, H, W, H])
        boxlast = boxxyxy * scale
        return boxlast, label, conf


from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def draw(X, boxlast, label, conf, targets=None, save_path=""):
    image = pil_to_tensor(X)

    labels = [f"{int(c)}: {float(s):.2f}" for c, s in zip(label, conf)]

    result = draw_bounding_boxes(
        image, boxlast.cpu(), labels=labels, colors="red", width=3
    )

    if targets is not None:
        result = draw_bounding_boxes(
            result,
            targets[0]["gtbox"].cpu(),
            labels=[str(int(c)) for c in targets[0]["cls"]],
            colors="green",
            width=2,
        )

    to_pil_image(result).save(save_path)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        # 1. 保存图片目录、标签目录
        # 2. 收集所有图片路径
        self.image_dir = os.path.join(data_dir, "images", "train2017")
        self.label_dir = os.path.join(data_dir, "labels", "train2017")
        self.files = os.listdir(self.image_dir)
        for filename in self.files.copy():
            # 1. 获取不带扩展名的文件名
            stem = os.path.splitext(filename)[0]

            # 2. 拼接对应的标签路径
            label_path = os.path.join(self.label_dir, stem + ".txt")

            # 3. 如果标签不存在，就删除这张图片
            if not os.path.isfile(label_path):
                self.files.remove(filename)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((640, 640)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        # 返回图片数量
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.files[idx])
        # 1. 读取第 idx 张图片
        img = Image.open(img_path).convert("RGB")
        # 2. 将图片 Resize 到统一尺寸
        # 3. 转换成 [3,H,W] Tensor
        img = self.transform(img)
        # 4. 读取对应的 YOLO .txt 标签
        stem = os.path.splitext(self.files[idx])[0]
        label_path = os.path.join(self.label_dir, stem + ".txt")

        target = {}
        cls = []
        boxs = []
        with open(label_path, "r") as f:
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
        return img, target


if __name__ == "__main__":
    # mode = model()
    # X = torch.randn([1, 3, 640, 640])
    # gtbox = torch.tensor(
    #     [[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.2, 0.2]],
    #     dtype=torch.float32,
    # )
    # clss = torch.tensor([1, 2, 3], dtype=torch.long)
    # targets = [{"cls": clss, "gtbox": gtbox}]  # construct targets
    # optimizer = torch.optim.Adam(mode.parameters(), lr=1e-4)
    # epoch = 100
    # for i in range(epoch):
    #     train_one_epoch(optimizer, mode, X, targets)
    # mode.eval()
    # with torch.no_grad():
    #     # out intension is the trained model that give it pre_X,get the prebox,and indices,compute the error with mathed gtbox
    #     cls, box = mode(X)
    #     indices = Getmatcher(cls, box, targets)
    #     pred_idex, gt_idx = indices[0]
    #     print(pred_idex)
    #     print(gt_idx)
    #     print(box[0, pred_idex])
    #     print(gtbox[gt_idx])
    #     error = torch.abs(box[0, pred_idex] - gtbox[gt_idx]).mean()
    #     gtxyxy = box_convert(gtbox[gt_idx], in_fmt="cxcywh", out_fmt="xyxy")
    #     prexyxy = box_convert(box[0, pred_idex], in_fmt="cxcywh", out_fmt="xyxy")
    #     giou = generalized_box_iou(prexyxy, gtxyxy)
    #     gious = torch.diag(giou).mean()
    #     print(error)
    #     print(gious)
    #     prob = cls.softmax(dim=-1)
    #     pred_classes = prob.argmax(dim=-1)
    #     print(pred_classes[0])
    #     print(pred_classes[0, pred_idex], clss[gt_idx])
    from torch.utils.data import random_split

    dataset = MyDataset("/opt/data/private/datasets/yangzhuohan/coco128")

    val_num = round(len(dataset) * 0.2)
    train_num = len(dataset) - val_num

    train_dataset, val_dataset = random_split(
        dataset, [train_num, val_num], generator=torch.Generator().manual_seed(42)
    )
    # img, target = dataset[0]
    # print(img.shape)
    # print(target["cls"].shape)
    # print(target["gtbox"].shape)
    # print(target["cls"].dtype)
    # print(target["gtbox"].dtype)

    train_dataloder = torch.utils.data.DataLoader(
        train_dataset, 6, True, collate_fn=collate_fn
    )
    val_dataloder = torch.utils.data.DataLoader(
        val_dataset, 6, False, collate_fn=collate_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    epochs = 400
    mode = model().to(device)
    mode.backbone.requires_grad_(False)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, mode.parameters()), lr=1e-4
    )
    minloss = 1e9
    # -----------------------
    print("---------------Start Train--------------------")
    for epoch in range(epochs):
        mode.train()
        mode.backbone.eval()
        print(f"-----------The Epoch={epoch+1}-----------------------")
        for i, (image, targets) in enumerate(train_dataloder):
            image = image.to(device)
            for target in targets:
                target["cls"] = target["cls"].to(device)
                target["gtbox"] = target["gtbox"].to(device)
            loss = train_one_epoch(optimizer, mode, image, targets)
        print("---------------Validation--------------------")
        mode.eval()
        with torch.no_grad():

            for image, targets in val_dataloder:
                image = image.to(device)
                for target in targets:
                    target["cls"] = target["cls"].to(device)
                    target["gtbox"] = target["gtbox"].to(device)
                cls, box = mode(image)
                indices = Getmatcher(cls, box, targets)
                loss = Loss(indices, targets, cls, box)

                valloss += loss.item()

        if epoch == epochs - 1:
            torch.save(mode.state_dict(), "overfitDetr.pt")

    # -----------------------
    # for epoch in range(epochs):
    #     mode.train()
    #     for i, (image, targets) in enumerate(train_dataloder):
    #         if i == 0:
    #             loss = train_one_epoch(optimizer, mode, image, targets)
    #             if minloss > loss.item():
    #                 minloss = loss.item()
    #                 torch.save(mode.state_dict(), "overfit.pt"7)
    #             break
    # for i, (image, targets) in enumerate(train_dataloder):
    #     if i == 0:
    #         mode = model()

    #         state = torch.load("overfit.pt")
    #         mode.load_state_dict(state)
    #         mode.eval()
    #         with torch.no_grad():
    #             cls, box = mode(image)
    #             # cls like 1 50 81 ,box like 1 50 4
    #             p = torch.softmax(cls[0], -1)
    #             conf, label = p.max(-1)
    #             print(f"conf is {conf}")
    #             print(f"label is {label}")
    #             keep = (conf > 0.8) & (
    #                 label != 80
    #             )  # 背景类 80 排除掉    threshold = 0.4
    #             box = box[0][keep]  # [N, 4] 归一化 cxcywh
    #             label = label[keep]
    #             conf = conf[keep]
    #             print(f"conf(threshold) is {conf}")
    #             print(f"label(threshold) is {label}")
    #             print(f"the really target:{targets[0]["cls"]}")
    #             # val时候，不存在targets，也就没有匈牙利匹配的，
    #             # 目标的顺序刚好对的上，看看giou
    #             boxyxy = box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
    #             reallyxyxy = box_convert(
    #                 targets[0]["gtbox"], in_fmt="cxcywh", out_fmt="xyxy"
    #             )
    #             giou = torch.diag(generalized_box_iou(boxyxy, reallyxyxy))
    #             H, W = image.shape[-2:]  # image: [1,3,640,640]
    #             scale = box.new_tensor([W, H, W, H])
    #             boxxyxy = boxyxy * scale
    #             reallyxyxy = reallyxyxy * scale
    #             print(giou)

    #             img = image[0]
    #             mean = img.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #             std = img.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #             img = img * std + mean
    #             # 返归一化
    #             img = (img.clamp(0, 1) * 255).to(torch.uint8).cpu()

    #             prenames = [
    #                 f"{int(c)}: {float(s):.2f}" for c, s in zip(label, conf)
    #             ]  # 把两个序列按位置配对，返回一组元组。
    #             realnames = [f"{int(c)}" for c in targets[0]["cls"]]
    #             paint = draw_bounding_boxes(img, boxxyxy, labels=prenames, colors="red")
    #             paint = draw_bounding_boxes(
    #                 paint, reallyxyxy, labels=realnames, colors="green"
    #             )
    #             to_pil_image(paint).show()
