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

7.**softmax activation function** for multi class classification in the final output layer of the fully connected layer.

8.**sigmoid activation function** for binary classification in the final output layer of the fully connected layer.

## Limitation of CNN
- Don't work well when multiple objects are in the image and draw bounding boxes around all the different objects.

## Region based CNN- R-CNN
- used for classification as well as objection detection with bounding boxes for multiple objects present in an image
- uses **selective search algorithm** for object detection to generate **region proposals**.

### Selective search in identifying multiple objects in an image
Step 1: **Generate initial sub-segmentation**. We generate as many regions, each of which belongs to at most one object.

Step 2: **Recursively combine similar regions into larger ones.** Here we use Greedy algorithm.
- From the set of regions, choose two regions that are most similar.
- Combine them into a single, larger region
- Repeat until only one region remains.

![image](https://user-images.githubusercontent.com/77944932/165237505-d6491ac0-5f2b-4777-ae04-e4a585c6ff24.png)

Step 3: Use the generated regions to produce candidate object locations.

### What is Region Proposal?
- set of **candidate detection** available to the detector. CNN runs the sliding windows over the entire image however R-CNN instead select just a few windows. R-CNN uses **2000 regions** for an image.
1.Generate category-independent region proposals using selective search to extract around 2000 region proposals. Warp each proposal.

2.Warped region proposals are fed to a large convolutional neural network. CNN acts as a feature extractor that extracts a fixed-length feature vector from each region. After passing through the CNN, R-CNN extracts a 4096-dimensional feature vector for each region proposal

3.Apply **SVM(Support Vector Machine)** to the extracted features from CNN. SVM helps to classify the presence of the object in the region. Regressor is used to predict the four values of the bounding box.

4.To all scored regions in an image, apply a greedy non-maximum suppression. **Non-Max suppression** rejects a region if it has an **intersection-over union (IoU)** overlap with a higher scoring selected region larger than a learned threshold.

![image](https://user-images.githubusercontent.com/77944932/165238138-5d116c79-3a20-4b26-96a2-ebbf1d71b285.png)

### What is greedy Non-Max suppression and why do we use it?
Our objective with object detection is to detect an object just once with one bounding box. However, with object detection, we may find multiple detections for the same objects. **Non-Max suppression ensures detection of an object only once.**

### Intersection over Union — IoU
IoU computes intersection over the union of the two bounding boxes, the bounding box for the ground truth and the bounding box for the predicted box by algorithm.

### Non-Max Suppression
- Non-Max Suppression will remove all bounding boxes where IoU is less than or equal to 0.5
- Pick the bounding box with the highest value for IoU and suppress the other bounding boxes for identifying the same object

### Limitation of R-CNN
- training is slow and expensive as extract 2000 regions for every image based on selective search.
- Extracting features using CNN for every image region. For N images, we will have N*2000 CNN features.
- R-CNN’s Object detection uses three models:
  •CNN for feature extraction
  •Linear SVM classifier for identifying objects
  •Regression model for tightening the bounding boxes
  
  ![image](https://user-images.githubusercontent.com/77944932/165238583-0a639d80-0099-43c6-88bd-76da7fb0dc75.png)
  
  ## Fast R-CNN
- Have one deep ConvNet to process the image once instead of 2,000 ConvNets for each regions of the image.
- Have one single model for extracting features, classification and generating bounding boxes unlike R-CNN that uses three different models

![image](https://user-images.githubusercontent.com/77944932/165239043-713059d8-8938-45e6-8989-bb75ac2251b3.png)

- Fast R-CNN network takes image and a set of object proposals as an input.
- Fast R-CNN uses a single deep ConvNet to extract features for the entire image once.
- **Region of interest (RoI)** layer is created to  extracts a fixed-length feature vector from the feature map for each object proposal for object detection.
- Fully Connected layers(FC) needs fixed-size input. Hence we use ROI Pooling layer to warp the patches of the feature maps for object detection to a fixed size.
- **ROI pooling** layer is then fed into the FC for **classification** as well as **localization**. RoI pooling layer uses **max pooling**. It **converts features inside any valid region of interest into a small feature map**.
- Fully connected layer branches into two sibling output layers:
-  One with softmax probability estimates over K object classes plus a catch-all “background” class
-  Another layer with a regressor to output four real-valued numbers for refined bounding box position for each of the K object classes.

### Key differences between R-CNN and Fast R-CNN
-Fast R-CNN uses **single Deep ConvNet for feature extractions**. A single deep ConvNet speeds us the image processing significantly unlike R-CNN that uses 2000 ConvNets for each region of the image.

-Fast R-CNN uses **softmax for object classification instead of SVM used in R-CNN**. Softmax slightly outperforming SVM for objection classification

-Fast R-CNN uses **multi task loss** to achieve an end to end training of Deep ConvNets **increases the detection accuracy**.

![image](https://user-images.githubusercontent.com/77944932/165239979-0915f5d0-494d-4d76-b357-538e9059be25.png)

### Limitation of Fast R-CNN
- usesselective search as a proposal method to find the Regions of Interest, which is slow and time consuming process. Not suitable for large real-life data sets.

## Faster R-CNN
- does not use expensive selective search instead uses Region Proposal Network.

### Faster R-CNN consists of two stages

- **First stage** is the deep fully convolutional network that proposes regions called a Region Proposal Network(RPN). RPN module serves as the attention of the unified network
- The second stage is the Fast R-CNN detector that **extracts features using RoIPool from each candidate box and performs classification and bounding-box regression**

![image](https://user-images.githubusercontent.com/77944932/165240571-dd50dae7-1468-4c67-8533-d22043a6964b.png)

### Region Proposal Network(RPN)
- Region Proposal Network takes an image of any size as input and outputs a set of rectangular object proposals each with an objectness score. It does this by sliding a small network over the feature map generated by the convolutional layer.
- Feature generated from RPN is fed into two sibling fully connected layers ,a box-regression layer for the bounding box and a box-classification layer for object classification.
- RPN is efficient and **processes 10 ms per image** to generate the ROI’s

![image](https://user-images.githubusercontent.com/77944932/165240932-839857b2-eb5c-4ba7-adf1-998e20a236bf.png)

### Anchors
- An anchor is centered at the sliding window in question and is associated with a scale and aspect ratio. Faster R-CNN uses 3 scales and 3 aspect ratio, yielding 9 anchors at each sliding windows.

- help with translational invariance.

- At each sliding window location, we simultaneously predict multiple region proposals. The number of maximum possible proposals for each location is denoted as k.Reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate the probability of object or not object for each proposal.

![image](https://user-images.githubusercontent.com/77944932/165241214-4c9343fc-697a-49d6-a903-f00e4683612f.png)

### Faster R-CNN

1.Feature Network which generates feature maps from the input image using deep convolutional layer(feature network).

2.Region Proposal Network (RPN) is used to identify different regions which uses 9 anchors for each sliding window. This helps with translational invariance. RPN **generate a number of bounding boxes called Region of Interests ( ROIs) with a high probability for the presence of an object**.

3.Detection Network is the R-CNN which takes input as the feature maps from the convolutional layer and the RPN network. This generates the bounding boxes and the class of the object.

-Faster R-CNN takes image as an input and is passed through the Feature network to generate the feature map.

-RPN uses the feature map from the Feature network as an input to generate the rectangular boxes of object proposals and the objectness score.

-The predicted region proposals from RPN are then reshaped using a RoI pooling layer. Warped into a fixed vector size.

-Warped fixed-size vector is then fed into two sibling fully connected layers, a regression layer to predict the offset values for the bounding box and a classification layer for object classification.

### Summary

We started with a simple CNN used for image classification and object detection for a single object in the image.

R-CNN is used for image classification as well as localization for multiple objects in an image.

R-CNN was slow and expensive so Fast R-CNN was developed as a fast and more efficient algorithm. Both R-CNN and Fast R-CNN used selective search to come up with regions in an image.

Faster R-CNN used RPN(Region Proposal Network) along with Fast R-CNN for multiple image classification, detection and segmentation.

## YOLO - You Only Look Once
- A single convolutional network simultaneously **predicts multiple bounding boxes and class probabilities for those boxes.**

![image](https://user-images.githubusercontent.com/77944932/165242336-500f7734-c161-4660-85c2-94b1f0421ca3.png)



![image](https://user-images.githubusercontent.com/77944932/165242414-d978d1c5-5b4c-4582-a1d7-6df11bfd7df3.png)

### YOLO v1
-  inspired by the GoogLeNet model for image classification.
-  has 24 convolutional layers followed by 2 fully connected layers.
-  YOLO uses a linear activation function for the final layer and a leaky ReLU for all other layers.
-  YOLO predicts the coordinates of bounding boxes directly using fully connected layers on top of the convolutional feature extractor. YOLO only predicts 98 boxes per image.
-  Limitatins : each grid cell only predicts two boxes and can only have one class and this limits the number of nearby objects that the model can predict. struggles with small objects that appear in groups, such as flocks of birds

### Fast YOLO / YOLO v2
- uses 9 convolutional layer instead of 24 used in YOLO and also uses fever filters.
- **Input size in YOLOv2 has been increased from 224*224 to 448*448.**
- YOLOv2 divides the entire image into 13 x 13 grid. This helps address the issue of smaller object detection in YOLO v1.
- YOLOv2 uses batch normalization which leads to significant improvements in convergence and eliminates the need for other forms of regularization. 
- YOLOv2 runs **k-means clustering on the dimensions of bounding boxes** to get good priors or anchors for the model. YOLOv2 found k= 5 gives a good tradeoff for recall vs. complexity of the model. YOLOv2 uses 5 anchor boxes
- YOLOv2 uses **Darknet architecture with 19 convolutional layers, 5 max pooling layers and a softmax layer for classification objects**.
- YOLOv2 use **anchor boxes to detect multiple objects, objects of different scales, and overlapping objects. This improves the speed and efficiency for object detection.**

### YOLO v3
-Uses 9 anchors
 
-Uses logistic regression to predict the objectiveness score instead of Softmax function used in YOLO v2

-YOLO v3 uses the Darknet-53 network for feature extractor which has 53 convolutional layers

## Mask R-CNN
- Mask R-CNN extends Faster R-CNN.

### What’s different in Mask R-CNN and Faster R-CNN?

1.Faster R-CNN has two outputs
-For each candidate object, a class label and a bounding-box offset;

2.Mask R-CNN has three outputs
-For each candidate object, a class label and a bounding-box offset;
Third output is the object mask

### What’s similar between Mask R-CNN and Faster R-CNN?

- Both Mask R-CNN and Faster R-CNN have a branch for classification and bounding box regression.
- Both use ResNet 101 architecture to extract features from image.
- Both use Region Proposal Network(RPN) to generate Region of Interests(RoI)

### How does Mask R-CNN work?

Mask R-CNN model is divided into two parts
1.Region proposal network (RPN) to proposes candidate object bounding boxes.

2.Binary mask classifier to generate mask for every class.

3.Image is run through the CNN to generate the feature maps.

4.Region Proposal Network(RPN) uses a CNN to generate the multiple Region of Interest(RoI) using a lightweight binary classifier. It does this using 9 anchors boxes over the image. The classifier returns object/no-object scores. Non Max suppression is applied to Anchors with high objectness score.

5.The RoI Align network outputs multiple bounding boxes rather than a single definite one and warp them into a fixed dimension.

6.Warped features are then fed into fully connected layers to make classification using softmax and boundary box prediction is further refined using the regression model.

7.Warped features are also fed into Mask classifier, which consists of two CNN’s to output a binary mask for each RoI. Mask Classifier allows the network to generate masks for every class without competition among classes

![image](https://user-images.githubusercontent.com/77944932/165262704-2abf5bd7-a2df-464a-b6a4-560ae179cf9d.png)

To predict multiple objects or multiple instances of objects in an image, Mask R-CNN makes thousands of predictions. Final object detection is done by removing anchor boxes that belong to the background class and the remaining ones are filtered by their confidence score. We find the anchor boxes with IoU greater than 0.5. Anchor boxes with the greatest confidence score are selected using Non-Max suppression.

### Non-Max Suppression
- Non-Max Suppression will remove all bounding boxes where IoU is less than or equal to 0.5

- Pick the bounding box with the highest value for IoU and suppress the other bounding boxes for identifying the same object

