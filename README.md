## Robotic DNN Inference Project: AlexNet & GoogLeNet Inference Model deployed in real time systems using NVIDIA DIGITS Workflow & Embeded NVIDIA JETSON TX2.


[//]: # (Written by Navid Safaeian July 25, 2018)

[p1datasample]:docs/P1DataSamples.png
[p1data]:docs/P1Data.png
[alexnetp1]:docs/AlexNet-P1Data-model.png
[googlenetp1]:docs/GoogLeNet-P1Data-model.png
[eval-alexnetp1]:docs/evaluate-AlexNet-P1Data.png
[eval-googlenetp1]: docs/evaluate-GoogLeNet-P1Data.png
[mysamples1]:docs/myDataSamples1.png
[mysamples2]:docs/myDataSamples2.png
[mydata1]:docs/myDataSet1.png
[mydata2]:docs/myDataSet2.png
[alexnet]:docs/AlexNet-model.png
[googlenet]:docs/GoogLeNet-model.png
[models]:docs/models.png
[con-alexnet]:docs/confusionMatrix-AlexNet-model.png
[con-googlenet]:docs/confusionMatrix-GoogLeNet-model.png
[eval-alexnet]:docs/testEval-AlexNet-model.png
[eval-googlenet]:docs/testEval-GoogLeNet-model.png
[tradeoff]:docs/tradeoff-cnn.png

## Abstract
This project is split into two sections which has two classification data models using Deep Neural Network technology. The first being an inference project against supplied training and test data which is classifying the bottles, candy wrappers and nothing on a moving belt. The second one classified the pill bottles, gum packs or nothing with different backgrounds which is applied on a moving belt or a robot arm with different angle views. The image data are come from my own captured image data by Samsung (S7) phone with different angles and perspectives. The project used the two models: AlexNet, GoogLeNet. For both classifications, the best results were presented in this article.

## Introduction
Classification includes a broad range of decision-theoretic approaches to the identification of images (or
parts thereof). All classification algorithms are based on the assumption that the image in question depicts
one or more features (e.g., geometric parts in the case of a manufacturing classification system, or spectral
regions in the case of remote sensing) and that each of these features belongs to one of several distinct and
exclusive classes. The classes may be specified a priori by an analyst (as in supervised classification) or
automatically clustered (i.e. as in unsupervised classification) into sets of prototype classes, where the analyst
merely specifies the number of desired categories.
In this project used NVIDIA’s DIGITS workflow[1] to rapidly prototype ideas that can be deployed on the
"__NVIDIA Jetson TX2 Module__" in close to real time. The DIGITS will prototype classification networks, detection networks, segmentation networks!
There are two parts in the project:
1. P1 moving belt image classification part used P1 dataset pictures of candy boxes, bottles, and nothing
(empty conveyor belt).
2. Object image classification part used the image set including pill bottle, gum pack, and nothing (different bckground with no objects) with an application for any moving belts or robot arm with different angle views which Author collected from Samsung phone.

##  1.  Moving belt image class (P1 Dataset)

### Background / Formulation:

The P1 image dataset is stored in /data/P1/ directory. It include all images of bottles, candy wrappers and
no object on a conveyor belt passing under a camera. A swing arm is used to sort all right objects to correct
the bins depending on classifying results.
P1 dataset image example: 

![][p1datasample]


### Data Acquisition:
It were split to two sets: training and validation dataset. They are color image and size is 256 x 256. This
dataset is provided from Udacity RoboND.

![][p1data]

### Model creation:
#### a. AlexNet Model was built as:

![][alexnetp1]

#### b. GoogLeNet Model was built as:

![][googlenetp1]

### The parameter setting and evaluating results:
Training epoch for both model set to 5 and the rest of the parameters used as default.<br/>
Digits evaluating result for AlexNet Model as:

![][eval-alexnetp1]

and Digits evaluating result for GoogLeNet Model as:

![][eval-googlenetp1]

As above evaluating result by Nvidia Digits, we can realize both AlexNet and GoogLeNet models are at least 75 percent accuracy and an inference time of less than 10ms.<br/>
AlexNet model comes with an accuracy of 75.41% and nearly an inference time of 4ms. However, GoogLeNet model was evaluated with the same accuracy of 75.41% and nearly an inference time of 5ms.

## 2. Object image classification (my own Dataset)

### Background / Formulation:
The captured image folder includs three subfolders, pill bottle folder, gum packs folder and nothing folder including the images for three types of pill bottles,three types of gum packs and three types of backgrounds respectively.<br/> 
The captured image data samples as follows:

![][mysamples1]
#  
![][mysamples2]

### Data Acquisition:
The object images were taken from S7 phone as videos with mp4 format (different videos for different classes with different objects), and then the required frames (images) were extracted from the videos by using a written [python code with opencv package](Image_data_preprocessing.ipynb) . The Images with three classes (pill bottle, gum pack, and nothing) were split three training, validation and test parts, the color image size is 256 X 256. 

![][mydata1]
The dataset specifications can be seen more visible as follows:
![][mydata2]

Here, the number of images (samples) in each class:<br/>
- __Gum pack images (three types): 1201__
- __Pill Bottle images (three types): 1114__
- __Nothing (background with no object) images: 935__

Note: Nothing class images include different backgrounds i.e. dark plain, bright plain, and other different patterns. <br/>
The following table presents the database for the captured image data:

<p>
<table width="800"><tbody>
    <tr><th align="center"></th><th align="center">Training</th><th align="center">Validating</th><th align="center">Testing</th></tr>
    <tr><td align="left">Image number </td><td align="center">2438</td>
    <td align="center">779</td><td align="center">33</td></tr>    
    <tr><td align="left">Percentage </td><td align="center">75%</td>
    <td align="center">24%</td><td align="center">1%</td></tr> 
</tbody></table>
</p>

### Model Creation:
The two model, AlexNet and GoogLeNet are created for my own collected image data with three categories (pill bottle, gum pack, and nothing). Both model represent very good results against test set, GoogLeNet model with 96.96% correct classification and AlexNet model with 100% correct classification. the models and results as follows:<br/>

Use default AlexNet network and GoogLeNet network in Digits, no change in model itself, with 60 epochs and 40 epochs, respectively.

![][alexnet]
#  
![][googlenet]

### Results:
AlexNet and GoogLeNet models, both were built and tested against the test set with 33 samples from the image data set. Both model results almost the same. However, the AlexNet model has slightly better results! 

![][models]
#   

The confision matrix for AlexNet Model presents 100% Top-1 accuracy and Top-5 accuracy.<br/>  

![][con-alexnet]
#  

The confision matrix for GoogLeNet Model presents 96.97% Top-1 accuracy and 100% Top-5 accuracy.

![][con-googlenet]
#  

The Digits test result screen copy for AlexNet model as follows:

![][eval-alexnet]
#  

The Digits test result screen copy for GoogLeNet model as follows:

![][eval-googlenet]
#  

### Discussion
Each class images (pill bottle, gum pack and nothing or background with no object) which was extracted from a cellphone by using written code with opencv package. To better learning of the models, the image data was tried to capture with different angles and perspective views, and also capturing different objects from each class. The nothing class included different kind of backgrounds with no object from dark to bright, and plain to strip pattern.
Many captured images do not show full objects, and this is an advantage of the learning model for better recognition of the partial object images.<br/> 
Therefore, the learning model suits for different applications like moving belt image classification or any mobile visual perception with different position. The quantity of the image data set was enough and generating of the augmented images were avoided in this project.<br/>   
The image source quality is very important for Deep Learning training result, the manually checking was applying all these generated images to make sure all image classes has been well labeled.
To achieve the best result, the LeNet network was not used because only 28x28 image size and gray color
can be used. But AlexNet and GoogLeNet, both networks support 256x256 color image.
Both AlexNet and GoogLeNet have been tested for classification of the image data, AlexNet and GoogLeNet showed pretty the same accuracy. 
Those model were learning well with this image data set and since a very high inference time in AlexNet model compared with GoogLeNet model (following graph[2]), therefore, it would be better to choose the GoogLeNet model for a robotic inference project.
![][tradeoff]

### Conclusion
#### P1 Moving Belt Image Classification:
Both AlexNet and GoogLeNet models were used with P1 dataset provided by the lesson in moving belt
image classification, the results achieved the requirements(least 75 percent accuracy and
an inference time of less than 10 ms.)

#### Object Image classification (collected image data):
Using cell phone video camera with extracting proper frames (images) by using opencv package is a good and fast method to collect image data. This tool leads to capture the image data with pretty good quality and enough quantity for a transfer learning on CNN networks such as GoogLeNet or AlexNet.
Both AlexNet and GoogLeNet models were tested in different epochs and the optimization methods such as SGD and ADAM. The best result based on the collected data comes with SGD optimazation method and the epochs more than 50 in AlexNet model and the epochs more than 20 in GoogLeNet model. It was preferred to use the default batch for both models because of a better result. The GoogLeNet model was preferred compared with AlexNet model since much less inference time and then much high info density. 

### Future Works

1. Learning different CNN models for more object classifications such as different kinds of shopping items in the grocery stores.  
2. Implementing a learning model and testing object detection and segmentation implementation, and deploying themodel on Jetson TX2 board and testing them in real world environment.
3. Using Nvidia DIGITS system to build and learning model with the medical image data such as MRI images and testing object detection regarding some abnormalities in the brain.  

#### References:
[1] Nvidia, DIGITS workflow, 2018 [link](https://developer.nvidia.com/digits) 
