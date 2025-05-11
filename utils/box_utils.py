import torch
import math


class DefaultBoxGenerator:
    def __init__(self, feature_maps, image_size=300, aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]):
        self.feature_maps = feature_maps
        self.image_size = image_size
        self.aspect_ratios = aspect_ratios
        self.num_layers = len(feature_maps)

        self.min_scale = 0.2
        self.max_scale = 0.9
        self.steps = [image_size / f for f in feature_maps]
        self.scales = [self.min_scale + (self.max_scale - self.min_scale) * i / (self.num_layers - 1)
                       for i in range(self.num_layers)] + [1.0]

        self.default_boxes = self._generate()

    def _generate(self):
        default_boxes = []
        for k, f_k in enumerate(self.feature_maps):
            s_k = self.scales[k]
            s_k_plus = self.scales[k + 1]
            for i in range(f_k):
                for j in range(f_k):
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k
                    default_boxes.append([cx, cy, s_k, s_k])
                    s_k_prime = math.sqrt(s_k * s_k_plus)
                    default_boxes.append([cx, cy, s_k_prime, s_k_prime])
                    for ar in self.aspect_ratios[k]:
                        default_boxes.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                        default_boxes.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
        return torch.tensor(default_boxes, dtype=torch.float32)

    def get_default_boxes(self):
        return self.default_boxes


def iou(boxes1, boxes2):
    # 确保两个张量位于相同的设备上
    device = boxes1.device if boxes1.device == boxes2.device else boxes1.device
    boxes2 = boxes2.to(device)  # 将 boxes2 移动到与 boxes1 相同的设备上

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = torch.max(rb - lt, torch.zeros_like(rb))  # 防止负的宽度/高度
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union  # 计算IoU


def encode(gt_boxes, default_boxes):
    gt_cxcy = (gt_boxes[:, 2:] + gt_boxes[:, :2]) / 2
    gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
    db_cxcy = default_boxes[:, :2]
    db_wh = default_boxes[:, 2:]
    loc = torch.cat([
        (gt_cxcy - db_cxcy) / db_wh,
        torch.log(gt_wh / db_wh)
    ], dim=1)
    return loc


def decode(loc_preds, default_boxes):
    db_cxcy = default_boxes[:, :2]
    db_wh = default_boxes[:, 2:]
    box_cxcy = loc_preds[:, :2] * db_wh + db_cxcy
    box_wh = torch.exp(loc_preds[:, 2:]) * db_wh
    boxes = torch.cat([box_cxcy - box_wh / 2, box_cxcy + box_wh / 2], dim=1)
    return boxes


def match(gt_boxes, gt_labels, default_boxes, iou_threshold=0.5):
    num_defaults = default_boxes.size(0)
    device = default_boxes.device  # 获取默认框的设备
    gt_boxes = gt_boxes.to(device)  # 将gt_boxes移动到default_boxes所在的设备上

    db = torch.cat(
        [default_boxes[:, :2] - 0.5 * default_boxes[:, 2:], default_boxes[:, :2] + 0.5 * default_boxes[:, 2:]], dim=1)
    iou_mat = iou(db, gt_boxes)
    best_gt_iou, best_gt_idx = iou_mat.max(dim=1)

    loc_targets = encode(gt_boxes[best_gt_idx], default_boxes)
    cls_targets = gt_labels[best_gt_idx]
    cls_targets[best_gt_iou < iou_threshold] = 0
    return loc_targets, cls_targets
