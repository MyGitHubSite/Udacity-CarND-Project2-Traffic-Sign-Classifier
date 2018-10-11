# **Traffic Sign Recognition** 
---
### Writeup / README

Link to project code: [project code](Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34,799 images.
* The size of the validation set is: 4,410 images.
* The size of test set is: 12,630 images.
* The shape of a traffic sign image is: 32x32x3 pixels.
* The number of unique classes/labels in the data set is: 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data distributed by type of traffic sign.  Some signs are represented more than others.

![Training Data Histogram](/images/Training%20Data%20Distribution.png)

Here is a visualization of the validation data.  It shows that the validation data has a simlar distribution as the training data.

![Validation Data Histogram](/images/Validation%20Data%20Distribution.png)

### 3. Design and Test a Model Architecture

#### A. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![Processing Pipeline](/images/6-End%20of%20speed%20limit%20(80km-h)_processed.png)

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

The difference between the original data set and the augmented data set is the following ... 

![Training Data Augmented Distribution](/images/Training%20Data%20Augmented%20Distribution.png)

#### B. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

#Layers
    layer1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID') + b1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = tf.nn.relu(tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='VALID') + b2)
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer3 = flatten(layer2)
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
    layer4 = tf.nn.dropout(layer4, p_dropout)
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)
    layer5 = tf.nn.dropout(layer5, p_dropout)
    logits = tf.matmul(layer5, W6) + b6
    return logits
    
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|

#### C. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### D. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### Validation Accuracy:

| Pre-Processing | Augment | Dropout | Epochs | Rate |   6x16  |  32x16  |  64x32  |  128x32 |
|:--------------:|:-------:|:-------:|:------:|:----:|:-------:|:-------:|:-------:|:-------:| 
|  (1)   |  No    | 1.0   | 20   | 0.001 | 0.916 | 0.924 | 0.935 | 0.943 | 
|  (2)   |  No    | 1.0   | 20   | 0.001 | 0.942 | 0.954 | 0.966 | 0.953 | 
|  (3)   |  No    | 1.0   | 20   | 0.001 | 0.941 | 0.942 | 0.955 | 0.962 | 
|  (3)   |  No    | 1.0   | 50   | 0.001 | 0.957 | 0.964 | 0.972 |<b>0.975</b>| 
|  (3)   |  No    | 0.0   | 50   | 0.001 | 0.950 | 0.961 | 0.960 | 0.962 | 
|        |        |       |      |       |       |       |       |       | 
|  (1)   |  Yes   | 1.0   | 20   | 0.001 | 0.835 | 0.909 | 0.924 | 0.908 | 
|  (2)   |  Yes   | 1.0   | 20   | 0.001 | 0.909 | 0.930 | 0.948 | 0.931 | 
|  (3)   |  Yes   | 1.0   | 20   | 0.001 | 0.911 | 0.938 | 0.943 | 0.952 | 
|  (3)   |  Yes   | 1.0   | 50   | 0.001 | 0.934 | 0.948 | 0.956 | 0.951 | 
|  (3)   |  Yes   | 0.9   | 50   | 0.001 | 0.930 | 0.945 | 0.957 | 0.948 | 
|        |        |       |      |       |       |       |       |       | 
|  (3)   |  No    | 0.5   | 500  | 0.0005 |   -   |   -   |   -   | 0.974 | 

Pre-Processing Steps:  
(1) Grayscale, Normalize  
(2) Crop, Resize, Grayscale, Normalize  
(3) Gamma Correction (0.4), Crop (3,3), Resize (32,32), Grayscale, Normalize

| Pre-Processing | Augment | Dropout | Epochs | Rate |  128x32 |
|:--------------:|:-------:|:-------:|:------:|:----:|:-------:|
|  (3)   |  No    | 1.0   | 30   | 0.001 | 0.970 |

My final model results were:
* training set accuracy of: 1.000
* validation set accuracy of: 0.970 
* test set accuracy of: 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

![Train_Validation_Test_Accuracy](/images/Test Accuracy0.png)

![Test_Accuracy_By_Class](/images/Class Test Accuracy_0.png)

### 4. Test a Model on New Images

#### A. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Image1](/my_data/11_right_of_way_at_the_next_intersection_200x200.png) 
![Image2](/my_data/12_priority_road_200x200.png) 
![Image3](/my_data/18_general_caution_200x200.png) 
![Image4](/my_data/25_road_work_200x200.png) 
![Image5](/my_data/33_turn_right_ahead_200x200.png)

The first image might be difficult to classify because ...

#### B. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of Way at Next Intersection      		| Right of Way at Next Intersection   									| 
| Priority Road     			| Priority Road 										|
| General Caution					| General Caution											|
| Road Work	      		| Road Work					 				|
| Turn Right Ahead			| Turn Right Ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### C. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


