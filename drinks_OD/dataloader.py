import os
import json
from PIL import Image
import torch

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
            # if num_objs == 0: continue
            boxes = []
            labels = []
            area = []
            if num_objs > 0:
                for region in data['regions']:
                    box = region['shape_attributes']
                    xmin = box['x']
                    ymin = box['y']
                    xmax = box['x'] + box['width']
                    ymax = box['y'] + box['height']
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(region['region_attributes']['name']))
                    area.append(box['width'] * box['height'])
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                area = torch.as_tensor(area)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
                area = torch.zeros(0, dtype=torch.float32)

            image_id = torch.tensor([idx])
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

        return img, target

    def __len__(self):
        return len(self.imgs)