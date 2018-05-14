# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn_architecture.png "CNN Architecture"
[image2]: ./images/final_architecture.png "Final Architecture"
[image3]: ./images/center.jpg "Center"
[image4]: ./images/left.jpg "Left"
[image5]: ./images/right.jpg "Right"
[image6]: ./images/center_track_two.jpg "Center Track Two"
[image7]: ./images/model.png "Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Based on the courses in this project, I use an architecture published by the autonomous vehicle team at Nvidia. 

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use Nvidia cnn architecture.

I followed the content in Behavior Cloning project. At first, I used Lenet to train the model. The performance was not good. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

At first, I only collected one forward and backward track on first track. I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Therefore, I added some dropout and pooling layers to lenet. However, the model is not good enough.

Then based on the introduction of Nvidia cnn architecture, I modified my "model.py". According to the layout of each image, I used Cropping to deal with images and increase the performance of my model.

The final step was to run the simulator to see how well the car was driving around track one. I built a model which is able to drive around track one. However, it has a bad behavior around track two. Therefore, I captured two tracks around track two, which makes the model more general. Rebuild the model based on more general data.

At the end of the process, the vehicle is able to drive autonomously around the track one and two without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two different laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

Then I added  the left side and right sides of the road back to train the modeal. 

![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

![alt text][image6]

After the collection process, I then preprocessed this data by lambda layer to normalize the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At the end of "model.py", I added some code to visualize training and validation loss. The following figure shows the loss.

![alt text][image7]

There are two videos that show the behavior of vehicle on track one and two.
