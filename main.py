import torch
from torch.utils.data import DataLoader

from Dataset import COCODataset, collate_fn
from Encoder import DetrModel
from MatchCost import HungarianMatcher
from Loss import Loss


def train_one_epoch(model, data_loader, matcher, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        # 模型内部已有 Backbone、Encoder、Decoder 和两个预测头。
        x, pred_logits, pred_boxes = model(images)
        indices = matcher(pred_logits, pred_boxes, targets)
        loss = Loss(pred_logits, pred_boxes, targets, indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

    return total_loss / len(data_loader.dataset)


@torch.no_grad()
def evaluate(model, data_loader, matcher, device):
    model.eval()
    total_loss = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        # val 同样用 GT 匹配来算 loss，但不反向、不更新参数。
        x, pred_logits, pred_boxes = model(images)
        indices = matcher(pred_logits, pred_boxes, targets)
        loss = Loss(pred_logits, pred_boxes, targets, indices)
        total_loss += loss.item() * images.shape[0]

    return total_loss / len(data_loader.dataset)


if __name__ == "__main__":
    # 在项目目录运行 python main.py，改这里的数据路径和参数即可。
    data_root = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
    epochs = 50
    batch_size = 4
    image_size = 640
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = COCODataset(data_root, "train2017", image_size)
    val_dataset = COCODataset(data_root, "val2017", image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = DetrModel().to(device)
    matcher = HungarianMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_loss = float("inf")

    print(
        f"device={device}, train图片={len(train_dataset)}, val图片={len(val_dataset)}"
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, matcher, optimizer, device)
        val_loss = evaluate(model, val_loader, matcher, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"[Train] loss={train_loss:.4f}")
        print(f"[Val]   loss={val_loss:.4f}")

        # 只保存验证 loss 最低的模型参数。
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best.pth")
            print(f"保存 best.pth，val loss={best_loss:.4f}")
