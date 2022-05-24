import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vision_utils import utils
from vision_utils.engine import evaluate
from dataloader import DrinksDataset
from gdrive_downloader import download_dataset, download_model

if __name__ == '__main__':
    data_path = "drinks"
    download_dataset(data_path)
    test_annotations_json = "drinks/labels_test.json"
    test_dataset = DrinksDataset(data_path, test_annotations_json, T.ToTensor())

    test_dataloader = DataLoader(test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True,
                          collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 4             # 3 drinks + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model_path = 'fasterrcnn_model_drinks_Epoch9.pt'      # edit epoch as needed
    download_model(model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    evaluate(model, test_dataloader, device=device)
