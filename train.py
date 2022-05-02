import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
from engine import train_one_epoch
from dataloader import DrinksDataset
from google_downloader import download_dataset

if __name__ == '__main__':
    data_path = "drinks"
    download_dataset(data_path)
    train_annotations_json = "drinks/labels_train.json"
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

    # model_path = 'fasterrcnn_model_drinks_Epoch7.pt'      # edit epoch as needed
    # download_model(model_path)

    # checkpoint = torch.load(model_path)
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

        PATH = f"fasterrcnn_model_drinks_Epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
        
        epoch += 1