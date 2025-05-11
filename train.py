
import torch
from torch.utils.data import DataLoader
from dataset.yolo_dataset import YoloDataset
from model.ssd_model import SSDResNet
from utils.loss import SSDLoss
from utils.box_utils import DefaultBoxGenerator, match


def collate_fn(batch):
    images, targets = zip(*batch)

    max_len = max(len(target) for target in targets)
    padded_targets = []

    for target in targets:

        padding = torch.zeros((max_len - len(target), 5))
        padded_target = torch.cat([target, padding], dim=0)
        padded_targets.append(padded_target)

    padded_targets = torch.stack(padded_targets, dim=0)


    return torch.stack(images, dim=0), padded_targets


def train():
    dataset = YoloDataset('./data')
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSDResNet(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SSDLoss()

    db_gen = DefaultBoxGenerator([38, 19, 10, 5, 3, 1])
    default_boxes = db_gen.get_default_boxes().to(device)

    for epoch in range(20):
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            loc_preds, cls_preds = model(imgs)

            loc_preds = torch.cat([l.view(l.size(0), -1, 4) for l in loc_preds], dim=1)
            cls_preds = torch.cat([c.view(c.size(0), -1, 2) for c in cls_preds], dim=1)

            loss_total = 0
            for b in range(imgs.size(0)):
                boxes = targets[b][:, :4] / 300
                labels = targets[b][:, 4].long()
                loc_t, cls_t = match(boxes, labels, default_boxes)
                loss = criterion(loc_preds[b], loc_t, cls_preds[b], cls_t)
                loss_total += loss

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss_total.item():.4f}")
