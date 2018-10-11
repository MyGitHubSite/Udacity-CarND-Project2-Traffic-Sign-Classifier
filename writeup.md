# **Traffic Sign Recognition** 
---
### Writeup / README

Link to project code: [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34,799 images.
* The size of the validation set is: 4,410 images.
* The size of test set is: 12,630 images.
* The shape of a traffic sign image is: 32 pixels x 32 pixels x 3 channels (RGB).
* The number of unique classes/labels in the data set is: 43.

#### 2. Include an exploratory visualization of the dataset.

Here are some random samples of one of the sign classes:

![Examples](/images/6-End%20of%20speed%20limit%20(80km-h).png)

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data is distributed by type of traffic sign.  Some signs are represented more than others.

![Training Data Histogram](/images/Training%20Data%20Distribution.png)

Here is a visualization of the validation data.  It shows that the validation data has a simlar distribution as the training data.

![Validation Data Histogram](/images/Validation%20Data%20Distribution.png)

### 3. Design and Test a Model Architecture

In order to improve the classifications I tried out various data pre-processing and augmentation techniques.

My data processing pipeline consisted of:
 - gamma correction (to increase the brightness of images)
 - cropping (to eliminate parts of the image not needed)
 - resizing (to make the images all 32x32 pixels again)
 - grayscaling (to eliminate the color channels which are not needed, also to simplify the images for training)
 - normalization (to mostly improve the calculation of gradients in the training)
 - rotation (to augment the data using variations of originals)
 - translation (to augment the data using variations of originals)

Here is a visualization of my processing pipeline:

![Processing Pipeline](/images/6-End%20of%20speed%20limit%20(80km-h)_processed.png)

Because some classes had less data than others the augmentation was an attempt to increase the number of images so that each class had about the same number of images.  (Note: Ultimately, I did not use the augmented data for my final model.)

Here is an example of the training data after the augmentation:

![Training Data Augmented Distribution](/images/Training%20Data%20Augmented%20Distribution.png)

#### B. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image						| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x128 	|
| RELU                  |												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x128 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Flatten               | outputs 800                                   |
| Fully connected		| outputs 120 									|
| RELU                  |                                               |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Softmax				| outputs 43   									|

#### C. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used various model acrchitectures and hyperparameters to find a "best" solution for validation accuracy.  In all cases I used a batch size of 128.  The results of my model runs for validation accuracy are summarized in the table below:

#### Validation Accuracy:

|  #  | Pre-Processing | Augment | Dropout | Epochs | Rate |   6x16  |  32x16  |  64x32  |  128x32 |
|:---:|:--------------:|:-------:|:-------:|:------:|:----:|:-------:|:-------:|:-------:|:-------:| 
|  1 |  (1)   |  No    | 1.0   | 20   | 0.001 | 0.916 | 0.924 | 0.935 | 0.943 | 
|  2 |  (2)   |  No    | 1.0   | 20   | 0.001 | 0.942 | 0.954 | 0.966 | 0.953 | 
|  3 |  (3)   |  No    | 1.0   | 20   | 0.001 | 0.941 | 0.942 | 0.955 | 0.962 | 
|  4 |  (3)   |  No    | 1.0   | 50   | 0.001 | 0.957 | 0.964 | 0.972 |<b>0.975</b>| 
|  5 |  (3)   |  No    | 0.9   | 50   | 0.001 | 0.950 | 0.961 | 0.960 | 0.962 | 
|    |        |        |       |      |       |       |       |       |       | 
|  6 |  (1)   |  Yes   | 1.0   | 20   | 0.001 | 0.835 | 0.909 | 0.924 | 0.908 | 
|  7 |  (2)   |  Yes   | 1.0   | 20   | 0.001 | 0.909 | 0.930 | 0.948 | 0.931 | 
|  8 |  (3)   |  Yes   | 1.0   | 20   | 0.001 | 0.911 | 0.938 | 0.943 | 0.952 | 
|  9 |  (3)   |  Yes   | 1.0   | 50   | 0.001 | 0.934 | 0.948 | 0.956 | 0.951 | 
| 10 |  (3)   |  Yes   | 0.9   | 50   | 0.001 | 0.930 | 0.945 | 0.957 | 0.948 | 
|    |        |        |       |      |       |       |       |       |       | 
| 11 |  (3)   |  No    | 0.5   | 500  | 0.0005 |   -   |   -   |   -   | 0.974 | 

Pre-Processing Steps:  
(1) Grayscale, Normalize  
(2) Crop(3,3), Resize(32,32), Grayscale, Normalize  
(3) Gamma Correction (0.4), Crop (3,3), Resize (32,32), Grayscale, Normalize

The pre-processing steps did improve the validation.  In the case of gamma correction it only improved validation when the depths of the first and second layer were 128 and 32 in my scenarios.  Augmenting the data did not improve things for me so utimately I did not use it.  For future improvements I would take another look at augmenting to see why things were not improving.

I tried dropout and changing the number of epochs.  Ultimately, I did not go too far down the path of using dropout.  The number of epochs did have an effect on validation accuracy and I finally settled on early stoppage of 30 epochs for my final model.

In the final entry in the table I went for a large run with 50% dropout and a lighly lower learning rate.  It did not beat one of my more simple runs so it was not used in my final model.

The architecture which gave me the best validation accuracy was model #4.

Here is a training, validation and test plot for model #4.

![Train_Validation_Test_Accuracy](/images/Training_Accuracy30.png)

For my final model I used the architecture of model #4 in the table but with only 30 epochs.  The final model results were:

* training set accuracy of: 1.000
* validation set accuracy of: 0.975 
* test set accuracy of: 0.952

I ran a test on each sign class to find strengths and weaknesses:

![Test_Accuracy_By_Class](/images/Class%20Test%20Accuracy_0.png)

For future improvements I would focus on increasing the number of quality samples for the classes which are performing poorly.

### 4. Test a Model on New Images

#### A. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![Image1](/my_data/11_right_of_way_at_the_next_intersection_200x200.png) 
![Image2](/my_data/12_priority_road_200x200.png) 
![Image3](/my_data/18_general_caution_200x200.png) 
![Image4](/my_data/25_road_work_200x200.png) 
![Image5](/my_data/33_turn_right_ahead_200x200.png)

The signs are centered well in the images but do contain extra information within them which could lead to some uncertainty.  Image #5 as an example has mutliple extra traffic signs in the background.

#### B. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of Way at Next Intersection      		| Right of Way at Next Intersection   									| 
| Priority Road     			| Priority Road 										|
| General Caution					| General Caution											|
| Road Work	      		| Road Work					 				|
| Turn Right Ahead			| Turn Right Ahead      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### C. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|100.000   11: Right-of-way at the next intersection    100.000%
|   30: Beware of ice/snow                       0.000%
|   10: No passing for vehicles over 3.5 metric tons 0.000%
|   16: Vehicles over 3.5 metric tons prohibited 0.000%
|   21: Double curve                             0.000%

True Label is: = 12: Speed limit (30km/h)          

   12: Priority road                            100.000%
   15: No vehicles                              0.000%
    9: No passing                               0.000%
   40: Roundabout mandatory                     0.000%
    7: Speed limit (100km/h)                    0.000%

True Label is: = 18: Speed limit (50km/h)          

   18: General caution                          100.000%
   26: Traffic signals                          0.000%
   25: Road work                                0.000%
   27: Pedestrians                              0.000%
   20: Dangerous curve to the right             0.000%

True Label is: = 25: Speed limit (60km/h)          

   25: Road work                                100.000%
   22: Bumpy road                               0.000%
   19: Dangerous curve to the left              0.000%
   30: Beware of ice/snow                       0.000%
   29: Bicycles crossing                        0.000%

True Label is: = 33: Speed limit (70km/h)          

   33: Turn right ahead                         100.000%
   39: Keep left                                0.000%
   13: Yield                                    0.000%
   37: Go straight or left                      0.000%
   12: Priority road                            0.000%
