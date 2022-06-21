import cv2
import numpy as np
import imutils
#%%
cap = cv2.VideoCapture('Video1.avi')

seq=[]

while True:
	ret_val, frame = cap.read()
	if frame is None:
		break
	seq.append(frame)

# convert to numpy array
seq = np.array(seq)
# read background of this video
background = cv2.imread('background1.jpg')
#%%
#for i in range(len(seq)//1):
#    # compute the difference between the current frame and
#	# background
#    frame = seq[i]
#    diff =cv2.absdiff(frame,background)
#    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#    #blur it
#    gray_diff = cv2.GaussianBlur(gray_diff, (3, 3), 0)
#    
#    
#    #thresholding for find moving objects
#    thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)[1]
#	# dilate the thresholded image to fill in holes, then find contours
#	# on thresholded image
##    kernel = np.ones([11,11])
##    thresh = cv2.dilate(thresh, kernel, iterations=2)
##    thresh = cv2.erode(thresh, kernel, iterations=2)
#    kernel = np.ones((29,29),np.uint8)
#    filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#    
#    kernel = np.ones((9,9),np.uint8)
#    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
#    
#    cnts = cv2.findContours(filled.copy(), cv2.RETR_TREE,
#		cv2.CHAIN_APPROX_SIMPLE)
#    cnts = imutils.grab_contours(cnts)
#	# loop over the contours
#    for c in cnts:
#		# if the contour is too small, ignore it
#        if cv2.contourArea(c) < 50:
#            continue
#		# compute the bounding box for the contour, draw it on the frame,
#		# and update the text
#        (x, y, w, h) = cv2.boundingRect(c)
##        find color of moving object
#        bb = frame[y+h//4:y+3*h//4,x+w//4:x+3*w//4,:];
#        color = np.mean(bb,axis=0)
#        color = np.floor(np.mean(color, axis=0))
#        # just select middle point intensity
##        color = np.array([frame[y+h//2,x+w//2,0], frame[y+h//2,x+w//2,1], frame[y+h//2,x+w//2,2]],dtype='float')
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)    
#        cv2.rectangle(filled, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)    
#
#    cv2.imshow('thresh',frame)
#    cv2.waitKey(10)
#cv2.destroyAllWindows()
#%% optical flow
prev_gray = cv2.cvtColor(seq[0], cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (9, 9), 0)
# Create mask
mask = np.zeros_like(background)
# Set image saturation to maximum value as we do not need it
mask[:,:, 1] = 255
#    
for i in range(len(seq)//5-1):
    # compute the difference between the current frame and
	# background
    frame = seq[i+1]
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Calculate dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
     # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[:,:, 0], flow[:,:, 1])
     # Set image hue according to the optical flow direction
    mask[:,:,0] = angle * 180 / np.pi / 2
     # Set image value according to the optical flow magnitude (normalized)
    magnitude = cv2.threshold(magnitude, 10, 255, cv2.THRESH_TRUNC)[1]
    mask[:,:,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #mm = cv2.threshold(mag, 60, 255, cv2.THRESH_BINARY)[1]
     # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    cv2.imshow('Dense optical flow', dense_flow)
        
    prev_gray = gray
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
















