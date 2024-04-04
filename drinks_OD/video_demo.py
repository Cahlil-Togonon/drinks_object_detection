import cv2
import argparse
from PIL import Image

import torch
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from gdrive_downloader import download_model

class  VideoDemo():
    def __init__(self,
                model,
                camera=0,
                width=640,
                height=480,
                record=False,
                filename="demo.mp4",
                Transforms=None):

        self.model = model
        self.camera = camera
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.Transforms = Transforms
        self.videowriter = None
        self.colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 0)]
        self.class_names = ["background", "Water", "Coke", "Juice"]

        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            self.videowriter = cv2.VideoWriter(self.filename,
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,
                                                (self.width, self.height),
                                                isColor=True)

    def loop(self):
        font = cv2.FONT_HERSHEY_TRIPLEX

        while True:
            _, image = self.capture.read()

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).convert("RGB")
            if self.Transforms is not None:
                img = self.Transforms(img)
            img = img[None, :]
            img = img.to('cuda')

            output = self.model(img)

            labels = output[0]['labels']
            rects = output[0]['boxes']

            for i in range(len(labels)):
                rect = rects[i]
                x1 = int(rect[0])
                y1 = int(rect[1])
                x2 = int(rect[2])
                y2 = int(rect[3])
                
                index = labels[i]
                color = self.colors[index]
                class_name = self.class_names[index]

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image,
                            class_name,
                            (x1, y1-15),
                            font,
                            1,
                            color,
                            2)

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
    
    model_path = './fasterrcnn_model_drinks_Epoch9.pt'      # edit epoch as needed
    download_model(model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default=False, required=False)
    parser.add_argument("--filename", default="demo.mp4", required=False)
    args = parser.parse_args()

    videodemo = VideoDemo(model=model,
                        camera=4,
                        record=args.record,
                        filename=args.filename,
                        Transforms=T.ToTensor())
    videodemo.loop()