# Drinks Object Detection
Object Detection model using Torchvision's Faster RCNN on a Drinks Dataset.
- Adrian Cahlil Eiz G. Togonon (agtogonon@up.edu.ph)

# Set-up
1) Fork and Clone this Git repository
2) run `pip install -r requirements.txt`. Note that this repository only requires `Torch`, `Torchvision`, and `gdown`. Download Torch and Torchvision with cuda enambled if possible.

# Testing the pre-trained model
run `python test.py` at the terminal.
- It will automatically download the Drinks Dataset to `/drinks`, as well as the latest pre-trained model from Google Drive using `gdrive_downloader.py`.
- Pre-trained model `fasterrcnn_model_drinks_Epoch9.pt` is the model checkpoint of the training script after 10 epochs

# Training the model
run `python train.py` at the terminal.
- default number of epochs is 10 for the training

# Real-time Object Detection
You can also run a real-time tracking on your camera using `python video_demo.py`. It uses the latest pre-trained model mentioned above and shows where it detects a Summit Water Bottle, Coca-Cola Can, and Del Monte Pineapple Juice on screen.

# Drinks Dataset
The Drinks Dataset is comprised of ~1000 images of 3 drinks in various positions. The 3 drinks are Summit Water Bottle, Coca-Cola Can, and Del Monte Pineapple Juice.
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

# Faster RCNN
TBA

# Citations
TBA
