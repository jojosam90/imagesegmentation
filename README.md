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
- Both use **ResNet 101 architecture (Backbone Model)** to extract features maps from image.
- Both use **Region Proposal Network(RPN)** to generate Region of Interests(RoI).This basically predicts if an object is present in that region (or not). In this step, we get those regions or feature maps which the model predicts contain some object.
- The regions obtained from the RPN might be of different shapes.A pooling layer and convert all the regions to the same shape. Next, these regions are passed through a fully connected network so that the class label and bounding boxes are predicted.

### How does Mask R-CNN work?

![image](https://user-images.githubusercontent.com/77944932/165463214-b4484baf-fa18-457d-95e2-e86fbb26b35b.png)

Mask R-CNN model is divided into two parts

1.Region proposal network (RPN) to proposes candidate object bounding boxes.

2.Binary mask classifier to generate mask for every class.

3.Image is run through the CNN to generate the feature maps.

4.Region Proposal Network(RPN) uses a CNN to generate the multiple Region of Interest(RoI) using a lightweight binary classifier. The region that RPN scans over are called **anchors**. It does this using 9 anchors boxes over the image.The classifier returns object/no-object scores. Non Max suppression is applied to Anchors with high objectness score.

5.The RoI Align network outputs multiple bounding boxes rather than a single definite one and warp them into a fixed dimension.

6.Warped features are then fed into fully connected layers to make classification using softmax and boundary box prediction is further refined using the regression model.

7.Warped features are also fed into Mask classifier, which consists of two CNN’s to output a binary mask for each RoI. Mask Classifier allows the network to generate masks for every class without competition among classes

![image](https://user-images.githubusercontent.com/77944932/165262704-2abf5bd7-a2df-464a-b6a4-560ae179cf9d.png)

To predict multiple objects or multiple instances of objects in an image, Mask R-CNN makes thousands of predictions. Final object detection is done by removing anchor boxes that belong to the background class and the remaining ones are filtered by their confidence score. We find the anchor boxes with IoU greater than 0.5. Anchor boxes with the greatest confidence score are selected using Non-Max suppression.

### Non-Max Suppression
- Non-Max Suppression will remove all bounding boxes where IoU is less than or equal to 0.5

- Pick the bounding box with the highest value for IoU and suppress the other bounding boxes for identifying the same object

## Mask R-CNN Keywords

**Backbone Network — implemented as ResNet 101 and Feature Pyramid Network (FPN), this network extracts the initial feature maps which are forward propagated to other components.**

**Region Proposal Network(RPN)—is used to extract Region of Interest(ROI) from images and Non-max suppression is applied to select the most appropriate bounding boxes or ROI generated from RPN.**

**ROI Align — wraps Region of Interest(ROI) into fixed dimensions.**

**Fully Connected Layers — consists of two parallel layers, one uses softmax for classification and the other regression for bounding box prediction.**

**Mask Classifier — generates a binary mask for each instance in an image.**

**1.Backbone**

![image](https://user-images.githubusercontent.com/77944932/165269404-63b0aee0-7fee-4c75-9032-bd866ac1f7fd.png)

This is a standard convolutional neural network (typically, ResNet50 or ResNet101) that serves as a feature extractor. The early layers detect low level features (edges and corners), and later layers successively detect higher level features (car, person, sky).

**2.Region Proposal Network**
The regions that the RPN scans over are called **anchors**.
RPN doesn’t scan over the image directly (even though we draw the anchors on the image for illustration). Instead, the RPN scans over the backbone feature map.

The RPN generates two outputs for each anchor:

**Anchor Class**: One of two classes: foreground or background. The FG class implies that there is likely an object in that box.

**Bounding Box Refinement**: A foreground anchor (also called positive anchor) might not be centered perfectly over the object. So the RPN estimates a delta (% change in x, y, width, height) to refine the anchor box to fit the object better.

Using the RPN predictions, we pick the top anchors that are likely to contain objects and refine their location and size. If several anchors overlap too much, we keep the one with the highest foreground score and discard the rest (referred to as Non-max Suppression). 

### ROI Classifier & Bounding Box Regressor
This stage runs on the regions of interest (ROIs) proposed by the RPN. And just like the RPN, it generates two outputs for each ROI:

![image](https://user-images.githubusercontent.com/77944932/165270269-59314a29-e5f1-467f-ae1f-418bb1f89b9f.png)

**Class**: The class of the object in the ROI. Unlike the RPN, which has two classes (FG/BG), this network is deeper and has the capacity to classify regions to specific classes (person, car, chair, …etc.). It can also generate a background class, which causes the ROI to be discarded.

**Bounding Box Refinement**: Very similar to how it’s done in the RPN, and its purpose is to further refine the location and size of the bounding box to encapsulate the object.

### ROI Pooling
Classifiers don’t handle variable input size very well. They typically require a fixed input size. But, due to the bounding box refinement step in the RPN, the ROI boxes can have different sizes. That’s where ROI Pooling comes into play. **ROI pooling** refers to **cropping a part of a feature map and resizing it to a fixed size.**

![image](https://user-images.githubusercontent.com/77944932/165409705-8d547084-6e6f-4929-9c08-6085f4ff89b2.png)

A method named ROIAlign, in which they sample the feature map at different points and apply a bilinear interpolation.

### Segmentation Masks
- The mask branch is a convolutional network that takes the positive regions selected by the ROI classifier and generates masks for the regions.
- The generated masks are **low resolution: 28x28 pixels**. But they are **soft masks**, represented by **float numbers**, so they **hold more details than binary masks**
- The small mask size helps keep the mask branch light. During training, we scale down the ground-truth masks to 28x28 to compute the loss, and during inferencing we scale up the predicted masks to the size of the ROI bounding box and that gives us the final masks, one per object.

## Panoptic Segmentation

- Pan means “all” and optic means “vision”. Panoptic segmentation, therefore, roughly means “everything visible in a given visual field”.

- Panoptic segmentation can be broken down into three simple steps:

1) Separating each object in the image into individual parts, which are independent of each other.

2) Painting each separated part with a different color - labeling.

3) Classifying the objects into things and stuff.

**Things**

- refers to objects that have properly defined geometry and are countable, like a person, cars, animals, etc.

**Stuff**

- used to define objects that don’t have proper geometry but are heavily identified by the texture and material like the sky, road, water bodies, etc.

![image](https://user-images.githubusercontent.com/77944932/166875223-0fa428f3-ee0f-4d7e-bf91-11ac825baae3.png)

### How does Panoptic Segmentation work?

In panoptic segmentation, the input image is fed into two networks: a fully convolutional network (FCN) and Mask R-CNN.

- The FCN is responsible for capturing patterns from the uncountable objects - stuff – and it yields semantic segmentations.

- The FCN uses skip connections that enable it to reconstruct accurate segmentation boundaries. Also, skip connections enable the model to make local predictions that accurately define the global or the overall structure of the object.

- Likewise, the Mask R-CNN is responsible for capturing patterns of the objects that are countable - things - and it yields instance segmentations. It consists of two stages:

- Region Proposal Network (RPN): It is a process, where the network yields regions of interest (ROI).

- Faster R-CNN: It leverages ROI to perform classification and create bounding boxes.

- The output of both models is then combined to get a more general output.

However, this approach has several drawbacks such as:

- Computational inefficiency
- 
- Inability to learn useful patterns, which leads to inaccurate predictions
- 
- Inconsistency between the network outputs
- 
To address these issues, a new architecture called the Efficient Panoptic Segmentation or EfficientPS was proposed, which improves both the efficiency and the performance.

### EfficientPS

EfficientPS uses a shared backbone built on the architecture called the EfficientNet.

The architecture consists of:

1. EfficientNet: A backbone network for feature extraction. It also contains a two-way feature pyramid network that allows the bidirectional flow of information that produces high-quality panoptic results.

2. Two output branches: One for semantic segmentation and one for instance segmentation.

3. A fusion block that combines the outputs from both branch.

![image](https://user-images.githubusercontent.com/77944932/166875807-f1339652-80e6-4916-b7bd-19723ff702bf.png)

EfficientPS network is represented in red, while the two-way Feature Pyramid Network (FPN) is represented in purple, blue and green. The network for semantic and instance segmentation is represented in yellow and orange, respectively, while the fusion block is represented at the end.

The image is fed into the shared backbone, which is an encoder of the EfficientNet. This encoder is coupled with a two-way FPN that extracts a rich representation of information and fuses multi-scale features much more effectively.

The output from the EfficientNet is then fed into two heads in parallel: one for semantic segmentation and the other for instance segmentation.

The semantic head consists of three different modules, which enable it to capture fine features, along with long-range contextual dependencies, and improve object boundary refinement. This, in turn, allows it to separate different objects from each other with a high level of precision.

The instance head is similar to Mask R-CNN with certain modifications. This network is responsible for classification, object detection, and mask prediction.

The last part of the EfficientPS is the fusion module that fuses the prediction from both heads. 

This fusion module is not parameterized—it doesn’t optimize itself during the backpropagation. It is rather a block that performs fusion in two stages. 

In the first stage, the module obtains the corresponding class prediction, the confidence score bounding box, and mask logits. Then, the module:

1) Removes all the object instances with the confidence score lower than a threshold value.

2) Once reductant instances are removed, the remaining instances or mask-logits are resized followed by zero-padding.

3) Finally, the mask-logits are scaled the same resolution as the input image.

In the first stage, the network sorts the class prediction, bounding box, and mask-logits with respect to the confidence scores.

In the second stage, it is the overlapping of the mask-logit that is evaluated.

It is done by calculating the sigmoid of the mask-logits. Every mask-logit that has a threshold greater than 0.5, obtains a corresponding binary mask. Furthermore, if the overlapping threshold between the binary is greater than a certain threshold, it is retained, while the others are removed.

A similar thing is done for the output yielded from the semantic head. 

Once the segmentations from both heads are filtered, they are combined using the Hadamard product, and voila—we’ve just performed the panoptic segmentation.

![image](https://user-images.githubusercontent.com/77944932/166875971-15b63f2e-3229-4c62-bb81-aa3e3571f60b.png)

### Panoptic Segmentation applications

1) Medical Imaging

2) Autonomous vehicles

3) Digital Image processing

### Summary 

- Panoptic segmentation is an image segmentation task that combines the prediction from both instance and semantic segmentation into a general unified output.

-  Panoptic segmentation involves studying both stuff and things.

- The initial panoptic deep learning model used two networks: Fully convolutional network (FCN) for semantic segmentation and Mask R-CNN for instance segmentation which was slow and yielded inconsistent and inaccurate segmentations due to which EfficientPS was introduced.

- EfficientPS consists of a shared backbone that enables the network to efficiently encode and combine semantically rich multi-scale features. It is fast and consistent with the output.






