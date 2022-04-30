import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn
import utils

class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations_json, transforms=None):
        self.root = root
        self.transforms = transforms

        json_file = open(annotations_json)
        data_json = json.load(json_file)
        json_file.close()

        self.imgs = []
        self.targets = []
        idx = 0
        for keys in data_json["_via_img_metadata"]:
            data = data_json["_via_img_metadata"][keys]
            img_name = data['filename']
            img_path = os.path.join(self.root, img_name)
            img = Image.open(img_path).convert("RGB")
            self.imgs.append(img)

            num_objs = len(data['regions'])
            boxes = []
            labels = []
            area = 0
            for region in data['regions']:
                box = region['shape_attributes']
                xmin = box['x']
                ymin = box['y']
                xmax = box['x'] + box['width']
                ymax = box['y'] + box['height']
                area = box['width'] * box['height']
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(region['region_attributes']['name']))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = torch.tensor([area])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            self.targets.append(target)

            idx += 1

    def __getitem__(self, idx):
        img = self.imgs[idx]
        target = self.targets[idx]

        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    data_path = "../drinks"
    train_annotations_json = "../drinks/labels_train.json"
    train_dataset = DrinksDataset(data_path, train_annotations_json, transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    mrcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = mrcnn_model.roi_heads.box_predictor.cls_score.in_features
    mrcnn_model.roi_heads.box_predictor = maskrcnn_resnet50_fpn(in_features, num_classes)
    mrcnn_model.to(device)
    params = [p for p in mrcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #criterion = nn.CrossEntropyLoss()

    from engine import train_one_epoch
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(mrcnn_model, optimizer, train_dataloader, device, epoch, print_freq=10)
        lr_scheduler.step()
        #evaluate(model, data_loader_test, device=device)

    EPOCH = 10
    PATH = "mrcnn_model_drinks.pt"

    torch.save({
            'epoch': EPOCH,
            'model_state_dict': mrcnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)