# Drinks Object Detection
Object Detection using Torchvision's Faster-RCNN model on a Drinks Dataset.
- Adrian Cahlil Eiz G. Togonon (agtogonon@up.edu.ph)

# Set-up
1) Fork and clone this git repository.
2) run `pip install -r requirements.txt`. Note that this repository only requires `Torch`, `Torchvision`, and `gdown`. Download Torch and Torchvision with cuda-enabled if possible.

# Drinks Dataset
The Drinks Dataset is comprised of ~1000 images of at most 3 drinks in various positions. The 3 drinks are Summit Water Bottle, Coca-Cola Can, and Del Monte Pineapple Juice.
The Drinks Dataset also includes annotations and segmentation files in `.csv` and `.json` format.
`train.py` and `test.py` will automatically download and unpack the Drinks Dataset from `drinks.tar.gz` to the folder `/drinks`.
The directory format should be as follows:
```
path/to/this/repo/
            drinks/
                  image0001.jpg       # image files
                  ...
                  image1050.jpg
                  labels_test.json    # test dataset annotations file
                  labels_train.json   # train dataset annotations file
```

# Testing the pre-trained model
run `python test.py` at the terminal.
- It will automatically download the Drinks Dataset to `/drinks`, as well as the latest pre-trained model from Google Drive using `gdrive_downloader.py`.
- The pre-trained model `fasterrcnn_model_drinks_Epoch9.pt` is checkpoint of the training script after 10 epochs.
Here are the result of testing the pre-trained model (epoch=10):
```
Test:  [ 0/51]  eta: 0:07:13  model_time: 4.9107 (4.9107)  evaluator_time: 0.0100 (0.0100)  time: 8.5070  data: 3.5823  max mem: 835
Test:  [50/51]  eta: 0:00:00  model_time: 0.2240 (0.3161)  evaluator_time: 0.0010 (0.0015)  time: 0.2259  data: 0.0000  max mem: 835
Test: Total time: 0:00:20 (0.3924 s / it)
Averaged stats: model_time: 0.2240 (0.3161)  evaluator_time: 0.0010 (0.0015)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.893
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.863
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.895
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.838
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.919
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.919
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.863
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.922
```

# Training the model
run `python train.py` at the terminal.
- default number of epochs is 10 for the training. 
Results after training for 10 epochs:
```
Epoch: [9]  [   0/1001]  eta: 11:28:38  lr: 0.000005  loss: 0.0388 (0.0388)  loss_classifier: 0.0149 (0.0149)  loss_box_reg: 0.0232 (0.0232)  loss_objectness: 0.0002 (0.0002)  loss_rpn_box_reg: 0.0005 (0.0005)  time: 41.2769  data: 36.9610  max mem: 1444
Epoch: [9]  [ 100/1001]  eta: 0:14:28  lr: 0.000005  loss: 0.0407 (0.0422)  loss_classifier: 0.0108 (0.0133)  loss_box_reg: 0.0215 (0.0279)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0003 (0.0007)  time: 0.5760  data: 0.0001  max mem: 1444
Epoch: [9]  [ 200/1001]  eta: 0:10:19  lr: 0.000005  loss: 0.0374 (0.0434)  loss_classifier: 0.0134 (0.0143)  loss_box_reg: 0.0224 (0.0275)  loss_objectness: 0.0002 (0.0004)  loss_rpn_box_reg: 0.0004 (0.0012)  time: 0.5784  data: 0.0002  max mem: 1444
Epoch: [9]  [ 300/1001]  eta: 0:08:16  lr: 0.000005  loss: 0.0352 (0.0429)  loss_classifier: 0.0110 (0.0142)  loss_box_reg: 0.0234 (0.0273)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0007 (0.0011)  time: 0.5774  data: 0.0001  max mem: 1444
Epoch: [9]  [ 400/1001]  eta: 0:06:46  lr: 0.000005  loss: 0.0404 (0.0428)  loss_classifier: 0.0111 (0.0146)  loss_box_reg: 0.0220 (0.0270)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0005 (0.0010)  time: 0.5768  data: 0.0002  max mem: 1444
Epoch: [9]  [ 500/1001]  eta: 0:05:28  lr: 0.000005  loss: 0.0367 (0.0424)  loss_classifier: 0.0121 (0.0144)  loss_box_reg: 0.0219 (0.0268)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0004 (0.0009)  time: 0.5775  data: 0.0002  max mem: 1444
Epoch: [9]  [ 600/1001]  eta: 0:04:18  lr: 0.000005  loss: 0.0404 (0.0423)  loss_classifier: 0.0124 (0.0143)  loss_box_reg: 0.0271 (0.0268)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0006 (0.0009)  time: 0.5783  data: 0.0001  max mem: 1444
Epoch: [9]  [ 700/1001]  eta: 0:03:10  lr: 0.000005  loss: 0.0336 (0.0419)  loss_classifier: 0.0111 (0.0142)  loss_box_reg: 0.0201 (0.0265)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0006 (0.0009)  time: 0.5772  data: 0.0002  max mem: 1444
Epoch: [9]  [ 800/1001]  eta: 0:02:06  lr: 0.000005  loss: 0.0458 (0.0418)  loss_classifier: 0.0133 (0.0141)  loss_box_reg: 0.0271 (0.0265)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0005 (0.0010)  time: 0.5778  data: 0.0001  max mem: 1444
Epoch: [9]  [ 900/1001]  eta: 0:01:02  lr: 0.000005  loss: 0.0321 (0.0416)  loss_classifier: 0.0116 (0.0140)  loss_box_reg: 0.0182 (0.0264)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0005 (0.0010)  time: 0.5800  data: 0.0001  max mem: 1444
Epoch: [9]  [1000/1001]  eta: 0:00:00  lr: 0.000005  loss: 0.0293 (0.0415)  loss_classifier: 0.0118 (0.0140)  loss_box_reg: 0.0187 (0.0263)  loss_objectness: 0.0001 (0.0003)  loss_rpn_box_reg: 0.0003 (0.0010)  time: 0.5782  data: 0.0001  max mem: 1444
Epoch: [9] Total time: 0:10:20 (0.6194 s / it)
```

# Real-time Object Detection
You can also run a real-time tracking on your camera using `python video_demo.py`. 
It uses the pre-trained model mentioned above and should show bounding boxes and labels from the model's inference on screen.

# Faster RCNN
TBA

# Citations
TBA
