# Toy-Hulk-Detector

Notebook: Toy_Hulk_Detector.ipynb

## Prerequisites

Google Colab  
Tensorflow 2  
GPU < 5minutes 
object_detection.utils, PIL  
Numpy, Matplotlib  

## Installations

!pip install -U --pre tensorflow=="2.2.0"  
Clone the tensorflow models repository
Install the Object Detection API
Download the checkpoint and put it into models/research/object_detection/test_data/

## Problem Statement
The new release of Tensorflow Object Detection API comes with a notebook showing us how to fine-tune a RetinaNet pre-trained model to detect rubber duckies with only 5 images and <5 minutes of training time in Google Colab. In this notebook, I have made use of this model to detect the images of toy hulk that I have collected. There are 10 images of hulk for training the model.

As given in the %%html<a href="https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb">Official Tutorial</a> , first import all the needed libraries and load the images. For labelling the images, there are two cells provided in the official tutorial. One of them has hardcoded labels while the other shows of a new OD API utility for Google Colab that allows you to label images. I have made use of the second technique to do the labelling.

## Toy Hulk data
I have collected 10 images of a toy hulk for training. The coco dataset does not contain toy hulk (or other toy superheros), so this is a novel class. 19 images are collected for testing the model.

![alt text](https://github.com/sabdha/Toy-Hulk-Detector/blob/main/Screenshot%202021-02-19%20092646.png)

## Annotate images with bounding boxes

In this cell I have  annotated the toy hulks by drawing a box around the toy hulk in each image.   
Click `next image` to go to the next image and `submit` when there are no more images.
![alt text](https://github.com/sabdha/Toy-Hulk-Detector/blob/main/annotated.png)

## Data Preparation for training
The class annotations are added.Only one class is handled. Everything is converted to tensors, classes are converted to one-hot representations that the training loop expects. How the one-hot representation is done is better explained in the 'Toy_hulk_Detector.ipynb' present in this repository. 

## Model and Weights
As per the tutorial a single stage detection architecture (RetinaNet) is built and all but the classification layer at the top are restored (which will be automatically randomly initialized). A number of things in for the specific RetinaNet architecture at hand (including assuming that the image size will always be 640x640), however it is not difficult to generalize to other model configurations.

## Conclusion
The training takes very less time and the results are very impressive. This model can be extended to handle multiple classes.

Result:   
![alt text](https://github.com/sabdha/Toy-Hulk-Detector/blob/main/result1.jpg)  
![alt text](https://github.com/sabdha/Toy-Hulk-Detector/blob/main/result2.jpg)


## References
Most part of the code is taken from  
https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
