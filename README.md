# imagesegmentation
- deep learning model try to classify each image's pixel instead of whole image.
- deep learning model takes input image > based on class trained, try to classify each pixel into a class > output color coded so that easily distinguish one class from another.

![image](https://user-images.githubusercontent.com/77944932/165195778-9acaf4ce-ac35-4692-9afd-c6ab3ccf7e0b.png)

### VGG-16 Architecture

![image](https://user-images.githubusercontent.com/77944932/165211927-911f8e81-1a71-4e4e-846e-881ab13a96ad.png)

cross entropy loss as loss functions

## Types of Image Segmentation
### Semantic Segmentation
  - classify the objects belonging to the same class in the image with **single label**.
  
### Instance Segmentation
  - Combination of segmentation and object detection.

## Performance Evaluation Metrics for Image Segmentation in Deep Learning
### Pixel Accuracy
- ratio of pixels that classified / total number of pixels in image

![image](https://user-images.githubusercontent.com/77944932/165196648-bd457fb2-8052-4076-9912-57b0a59fc79c.png)

### Mean Pixel Accuracy
- ratio of correct pixels per class.

![image](https://user-images.githubusercontent.com/77944932/165196713-103911cf-9c3b-4c9c-8afa-0181b080311f.png)

### Intersection over Union (IoU) and Mean-IoU
- known as the **Jaccard Index** is used for both object detection and image segmentation.
- raction of area of intersection of the predicted segmentation of map and the ground truth map to the area of union of predicted and ground truth segmentation maps.

![image](https://user-images.githubusercontent.com/77944932/165196850-2dcfb801-f225-40de-a24d-8e2c15e151d5.png)

A - predicted 
B - ground truth 

### Dice Coefficient and Dice Loss
- ratio of the **twice the intersection of the predicted and ground truth** segmentation maps to the total area of both the segmentation maps.

![image](https://user-images.githubusercontent.com/77944932/165196982-6f41875b-8c25-44d9-b4fa-906a451323ea.png)

## Types of Image Segmentation Model 
### Fully Convolutional Networks (FCN)

![image](https://user-images.githubusercontent.com/77944932/165197207-729d2eff-d0ab-4316-af56-ae25cf6e47d2.png)

-  contains only convolutional layers
-  modified the GoogLeNet and VGG16 architectures by replacing the final fully connected layers with convolutional layers
-  input is an RGB image
-  hence its output a segmentation map of the input image instead of the standard **classification scores**.
-  Disadvantages : model was that it was very slow and could not be used for real-time segmentation

### SegNet

![image](https://user-images.githubusercontent.com/77944932/165197503-ee79c54b-e692-448f-bb6b-19b95d7de6d2.png)

- segmentation model based on the encoder-decoder architecture.
- contains only convolutional layers 
- Encoder contains 13 convolutional layers inside VGG16 network.
- Decoder contains upsampling layers and convolutional layers, responsible for the **pixel-wise classification** of the input image and **outputting the final segmentation map**.

### U-Net

![image](https://user-images.githubusercontent.com/77944932/165198719-4b6b07cf-ffdb-4bce-a7e0-7c5640051061.png)

- segmentation model based on the encoder-decoder architecture.
- mainly aimas at segmenting medical images.
- comprises of **two parts**. One is the **down-sampling network** part that is an FCN-like network. One is the **up-sampling network** that increase each dimensions in layers.

### Mask-RCNN

![image](https://user-images.githubusercontent.com/77944932/165199267-9c8b968b-7c37-4713-9425-3fe64342ecf0.png)

- contains 3 output branches.
-  branches for the bounding box coordinates, branches for output classes, and the branches for segmentation map.


## Real-Life Use Cases and Applications of Image Segmentation 
 - Medical Imaging
![image](https://user-images.githubusercontent.com/77944932/165199439-86468bf2-2a51-4426-a2cc-cca975d2c89f.png)

 - Autonomous Driving
 
 ![image](https://user-images.githubusercontent.com/77944932/165199809-bcc59295-2e03-47ca-be02-822a19aabebe.png)
 
 ![image](https://user-images.githubusercontent.com/77944932/165199829-2524c5f4-8051-4cfa-b8e6-b0a7f3ac7602.png)

 - Satellite Imaging
 
![image](https://user-images.githubusercontent.com/77944932/165199852-9cef67bd-efd8-41b8-a794-86ed63a7d130.png)

## How does image classification work using Convolution Neural Network(CNN)?

1.CNN takes input as an image “x”, which is a 2-D array of pixels with different color channels(Red,Green and Blue-RGB).

2.**Different filters or feature detector** applied to the input image to output **feature maps**.

3.Multiple convolutions are performed in parallel by applying **nonlinear function ReLU** to the convolutional layer. Feature detector identifies different things like edge detection, different shapes, bends or different colors etc.

![image](https://user-images.githubusercontent.com/77944932/165234924-404f2e90-0323-4ca0-8ada-8a9d3bdc311c.png)

4.Apply Min Pooling, Max Pooling or Average Pooling. **Max pooling** function provides better performance compared to min or average pooling.

5.Pooling helps with **Translational Invariance.** Translational invariance means that when we change the input by a small amount the pooled outputs does not change.Invariance of image implies that even when an image is rotated, sized differently or viewed in different illumination an object will be recognized as the same object.

6.Next,flatten the pooled layer to input it to a fully connected(FC) neural network.

7.







