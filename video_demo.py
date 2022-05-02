import cv2
import datetime
from PIL import Image

import torch
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from gdrive_downloader import download_model

class  VideoDemo():
    def __init__(self,
                 detector,
                 camera=0,
                 width=640,
                 height=480,
                 record=False,
                 filename="demo.mp4",
                 Transforms=None):
        self.camera = camera
        self.detector = detector
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.videowriter = None
        self.detector = detector
        self.Transforms = Transforms
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            self.videowriter = cv2.VideoWriter(self.filename,
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,
                                                (self.width, self.height), 
                                                isColor=True)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        line_type = 1

        while True:
            start_time = datetime.datetime.now()
            _, image = self.capture.read()

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# / 255.0
            img = Image.fromarray(img).convert("RGB")
            if self.Transforms is not None:
                img = self.Transforms(img)
            img = img[None, :]
            img = img.to('cuda')
            output = self.detector(img)
            class_names = output[0]['labels']
            rects = output[0]['boxes']

            items = {}
            for i in range(len(class_names)):
                rect = rects[i]
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2]
                y2 = rect[3]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                name = class_names[i]
                if name in items.keys():
                    items[name] += 1
                else:
                    items[name] = 1
                index = name.item()
                colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 0)]
                color = colors[index]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                labels = ["background", "Water", "Coke", "Juice"]
                cv2.putText(image,
                            labels[index],
                            (x1, y1-15),
                            font,
                            0.5,
                            color,
                            line_type)

            cv2.imshow('image', image)
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # close window with "q"
                break

        # if window is closed
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 4             # 3 drinks + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    
    model_path = 'fasterrcnn_model_drinks_Epoch9.pt'      # edit epoch as needed
    download_model(model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.to('cuda')
    model.eval()

    videodemo = VideoDemo(detector=model,
                            camera=0,
                            record=False,
                            filename='demo.mp4',
                            Transforms=T.ToTensor())
    videodemo.loop()