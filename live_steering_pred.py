import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import pickle
from scipy.misc import imread, imresize

from birdseye_view import BirdsEye_View
from lanefilter import LaneFilter
from curves_fit import Curves
from helpers import show_images, save_image, roi

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

smoothed_angle = 0

calibration_data = pickle.load(open("calibration_data.p", "rb" ))

matrix = calibration_data['camera_matrix']
distortion_coef = calibration_data['distortion_coefficient']
#source_points = [(680, 560), (305, 820), (1210, 820), (803, 560)]
source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
#destination_points = [(370, 260), (5, 520), (780, 390), (503, 260)]
#destination_points = [(680, 560), (305, 820), (1210, 820), (803, 560)]
destination_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
#source_points = [[0,0],[1920,0],[0,1080],[1920,1080]]
#destination_points = [(370, 260), (5, 520), (780, 390), (503, 260)]

p = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }

birdsEye = BirdsEye_View(source_points, destination_points, matrix, distortion_coef)
laneFilter = LaneFilter(p)
curves = Curves(number_of_windows = 20, margin = 100, minimum_pixels = 50, 
                ym_per_pix = 30 / 720 , xm_per_pix = 3.7 / 700)

def debug_pipeline(img):
    
  ground_img = birdsEye.undistort(img)
  birdseye_img = birdsEye.sky_view(img)
    
  binary_img = laneFilter.apply(img)
  sobel_img = birdsEye.sky_view(laneFilter.sobel_breakdown(ground_img))
  color_img = birdsEye.sky_view(laneFilter.color_breakdown(ground_img))
  
  wb = np.logical_and(birdsEye.sky_view(binary_img), roi(binary_img)).astype(np.uint8)
  result = curves.fit(wb)
    
  left_curve =  result['pixel_left_best_fit_curve']
  right_curve =  result['pixel_right_best_fit_curve']
    
  left_radius =  result['left_radius']
  right_radius =  result['right_radius']
  pos = result['vehicle_position_words']
  curve_debug_img = result['image']
  
  projected_img = birdsEye.project(ground_img, binary_img, left_curve, right_curve)
    
  return birdseye_img, sobel_img, color_img, curve_debug_img, projected_img, left_radius, right_radius, pos

def verbose_pipeline(img):
  b_img, s_img, co_img, cu_img, pro_img, lr, rr, pos = debug_pipeline(img)

  b_img = imresize(b_img, 0.25)
  s_img = imresize(s_img, 0.25)
  co_img = imresize(co_img, 0.25)
  cu_img = imresize(cu_img, 0.25)

  offset = [0, 320, 640, 960]
  #width, height = 480,270
  width, height = 320,180

  pro_img[:height, offset[0]: offset[0] + width] = b_img
  pro_img[:height, offset[1]: offset[1] + width] = co_img
  pro_img[:height, offset[2]: offset[2] + width] = s_img
  pro_img[:height, offset[3]: offset[3] + width] = cu_img

  text_pos = "vehicle pos: " + pos
  text_l = "left r: " + str(np.round(lr, 2)) 
  text_r = " right r: " + str(np.round(rr, 2))

  image = scipy.misc.imresize(img, [66, 200]) / 255.0
  degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    
  cv2.putText(pro_img, text_l, (20, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  cv2.putText(pro_img, text_r, (250, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  cv2.putText(pro_img, text_pos, (620, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  if float(lr)>float(rr):
      cv2.putText(pro_img, "<- Left", (530, 420), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
  #cv2.putText(pro_img, "Right ->", (530, 420), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
  else:
      cv2.putText(pro_img, "Right ->", (530, 420), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
  cv2.putText(pro_img, "%.2f" % degrees  , (530, 590), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
  return pro_img


cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    frame1 =  cv2.resize(frame,(1280,720), interpolation =cv2.INTER_AREA)

    cv2.imshow('Frame',verbose_pipeline(frame1))

    k=cv2.waitKey(5)
    if k == 27:
        break

cv2.destroyAllWindows()
cv2.release()
