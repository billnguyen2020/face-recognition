# Face Recognition using YOLO-Face and FaceNet

The Face Recognition application uses computer webcam to detect faces and classify among trained data. Codes in this project refer to various sources. See Reference section below.

## Architecture

![](https://i.imgur.com/b8ZzbTG.png)

### Face Detection

Video frames from webcam are processed using OpenCV. Faces in each frame are detected using YOLO-Face model. Face images are then cropped out and aligned to the center.

Output: `(160, 160, 3)` array for each face image

### Extract Embedding

FaceNet model is used to extract embedding from each detected face.

Output: Embedding vector with size `128` for each face image

### Face Recognition

Face images are classfied among trained data based on their embedding vector.

Output: Image class index

## Setup

1. Download `cfg`, `models` and `weights` folders [HERE (Google Drive)](https://drive.google.com/drive/folders/1nXQErNFchh8qFou-McBXwgQskrC8tlOe?usp=sharing) and put them in this repo directory.
2. Create directory `data`

```
mkdir data
```

The application directory should look like this:

```
face-recognition
├───cfg
├───data
├───models
├───weights
├───detect.py
├───helpers.py
└───...

```

3. Create new conda environment

```
conda env create -f environment.yml
conda activate face-recognition
```

## Collect training data

Use file `capture.py` to collect training data using webcam.

```
python capture.py
```

_Note:_
_- You should capture one person at a time_
_- You will be prompted to enter the class name. This should be the name of the person in front of the camera. Enter an existing name to add more data for the existing class or a new one to add more class._

Each class should have a minimum of 30 images for best model performance.

## Train Face Recognition Model

After you finish collecting data, run file `train.py` to train the classifier model.

```
python train.py
```

## Run Application

Start the application in file `detect.py` to test your model.

```
python detect.py
```

## Reference

- [Real time face recognition with CPU](https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5)

- [How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)
