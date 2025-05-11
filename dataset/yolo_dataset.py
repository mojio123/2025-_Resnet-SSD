import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def yolo_to_xyxy(parts, w, h):
    cls, xc, yc, bw, bh = map(float, parts)
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return int(cls), x1, y1, x2, y2

class YoloDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        self.transform = transform or T.Compose([
            T.Resize((300, 300)),
            T.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.splitext(img_path)[0] + '.txt'

        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)
                        boxes.append([x1, y1, x2, y2, cls])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        return self.transform(image), boxes

    def __len__(self):
        return len(self.img_files)
