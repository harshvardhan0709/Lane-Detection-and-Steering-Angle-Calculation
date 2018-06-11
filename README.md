# Lane-Detection-and-Steering-Angle-Calculation

#### Floating Ball Lane Detection 
Computer vision techniques to identify lane boundaries and compute the estimate the radius of curvature given a frame of video or live camera feed.\
&nbsp;&nbsp;&nbsp;&nbsp;**To achieve this, the following steps are taken:**
- Computed the camera calibration matrix and distortion coefficients of the camera lens used given a set of chessboard images   taken by the same camera 
- Used the aforementioned matrix and coefficient to correct the distortions given by the raw output from the camera 
- Use color transforms, and sobel algorithm to create a thresholded binary image that has been filtered out of unnecessary information on the image
- Apply perspective transform to see a “birds-eye view” of the image as if looking from the sky  
- Apply masking to get the region of interest, detect lane pixels,  
- Determine the best fit curve for each lane the curvature of the lanes 
- Project the lane boundaries back onto the undistorted image of the original view  
- Output a visual display of the lane boundaries and other related information  




![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/2.jpg "Logo Title Text 1")



![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/3.jpg "Logo Title Text 1")

 
 ![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/1.jpg "Logo Title Text 1")

 
 
 
 ![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/curve1.jpg "Logo Title Text 1")
 
 

![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/curve.jpg "Logo Title Text 1")




#### Steering Angle Calculation
Data.csv file contain two value Frame number and Steering Angle 

| Frame No.       | Steering Angle    |
| ----------------|:-----------------:|
| 1235            |       -5.43       |
| 1236            |       -6.81       |
| 1237            |       -7.29       |

###### Final Model Architecture

The final model architecture (NvidiaFull.py) consists of a convolution neural network with the following layers and layer sizes. The RGB input images are cropped and resized to 64x64 pixels outside the model.

To introduce nonlinearity I chose for all convolutional and fully connected layers a RELU activation. To avoid overfitting the model contains 3 dropout layers with a drop rate of 50% in the fully connected layers.

|  Layer 	     |  Depths/Neurons 	|  Kernel |	Activation |	Pool Size  |	Stride |	Border Mode |	Output Shape |	Params |
| -------------|:----------------:|:-------:|:----------:|:-----------:|:-------:|:------------:|:------------:|:-------:|
| Lambda 	     |						      |         |            |             |         |              |    3@64x6    |         |
| Convolution  |        3         |	  5x5   |	   RELU    |	           |	       |              |		 3@64x64 	 |   228   |
| Max pooling  | 		              |         |            |	   2x2     |	 2x2 	 |     same     |	   3@32x32 	 |         |
| Convolution  | 	     24         |	  5x5   |	   RELU    | 	           |	       |              |		24@32x32   |	1824   |
| Max pooling  | 		              |         |            |		 2x2     |	 2x2 	 |     same     | 	24@16x16   |         |	
| Convolution  | 	     36         |	  5x5   |	   RELU    | 	           |	       | 			        |	  36@16x16 	 |  21636  |
| Max pooling  |						      |         |            |		 2x2     |	 2x2   |	   same     |	   36@8x8    |         |	
| Convolution  | 	     48         |	  3x3 	|    RELU 	 |			       |         |              |    48@8x8	   |  15600  |
| Max pooling  | 			            |         |            |     2x2 	   |   1x1   |     same 	  |    48@4x4    |         | 	
| Convolution  | 	     64         |  	3x3 	|    RELU 	 | 	           |         |              |    64@4x4 	 |  27712  |
| Max pooling  | 		              |         |            |     2x2     |   1x1 	 |     same 	  |    64@2x2 	 |         |
| Flatten 		 | 					        |         |            |             |         |              |      256     |         |
| Fully connected|   	1164 	      |         | 	 RELU    |             |         |              |			 1164 	 |  299148 |
| Dropout 50%  |							    |         |            |             |         |              |      1164 	 |         |
| Fully connected| 	   100 		    |         |    RELU 	 |             |         |              |			 100     | 	116500 |
| Dropout 50%  |							    |         |            |             |         |              |      100 	   |         |
| Fully connected| 	   50 		    |         | 	 RELU    |             |         |              |			 50 	   |   5050  |
| Dropout 50%  |							    |         |            |             |         |              |      50 	   |         |
| Fully connected| 	   10 		    |         | 	 RELU    |             |         |              |			 10      |	 510   |
| Fully connected|    	1 				|         |            |             |         |              |       1 	   |    11   |


![alt text](https://github.com/harshvardhan0709/Lane-Detection-and-Steering-Angle-Calculation/blob/master/videos/Screenshot%20from%20output12345.mp4.png "Logo Title Text 1")


# Required library setup

run `pip install -r requirements.txt`

# How to Detect lane

Use `python live_lane_dect.py` to run the model on a live webcam feed

Use `python lane_dect_video.py` to apply lane detection system on video

the output video will be stored in Lane-Detection-and-Steering-Angle-Calculation folder

# How to Use Steering Angle Neural Network 

Use `python live_steering_pred.py` to run the model on a live webcam feed

Use `python steering_video.py` to run the model on video

#### Note : The Neural Network require lots of data to get train properly so that it can predict steering angle accurately.

# Video Output Links

<https://www.youtube.com/watch?v=LgjmPWEwYSo>\
<https://www.youtube.com/watch?v=9d-zSgkwHhE>

