import os
import json
import torch
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
from engine import train_one_epoch

class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations_json, Transforms=None):
        self.root = root
        self.Transforms = Transforms

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

            num_objs = len(data['regions'])
            if num_objs == 0: continue
            boxes = []
            labels = []
            area = []
            for region in data['regions']:
                box = region['shape_attributes']
                xmin = box['x']
                ymin = box['y']
                xmax = box['x'] + box['width']
                ymax = box['y'] + box['height']
                area.append(box['width'] * box['height'])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(region['region_attributes']['name']))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = torch.as_tensor(area)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            self.imgs.append(img)
            self.targets.append(target)
            idx += 1

    def __getitem__(self, idx):
        img = self.imgs[idx]
        target = self.targets[idx]

        if self.Transforms is not None:
            img = self.Transforms(img)
            #target = self.Transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    data_path = "../drinks"
    train_annotations_json = "../drinks/labels_train.json"
    train_dataset = DrinksDataset(data_path, train_annotations_json, T.ToTensor())

    train_dataloader = DataLoader(train_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True,
                          collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 4             # 3 drinks + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # checkpoint = torch.load("fasterrcnn_model_drinks_Epoch7.pt")      # edit epoch as needed
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    epoch = 0
    num_epochs = 10
    model.to(device)
    model.train()
    while epoch < num_epochs:
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        lr_scheduler.step()
        #evaluate(model, data_loader_test, device=device)

        PATH = f"fasterrcnn_model_drinks_Epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
        
        epoch += 1