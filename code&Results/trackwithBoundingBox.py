import cv2
import numpy as np
import imutils

background = cv2.imread('background1.jpg')
cap = cv2.VideoCapture('Video1.avi')
# track objects
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('moving_obj.mp4',fourcc, 30, (800,480))

while True:
    ret_val, frame = cap.read()
    if frame is None:
        break
    # compute the difference between the current frame and
	# background
    diff =cv2.absdiff(frame,background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #blur it
    gray_diff = cv2.GaussianBlur(gray_diff, (21, 21), 0)
    #thresholding for find moving objects
    thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
    kernel = np.ones([7,7])
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=2)
    cnts = cv2.findContours(erode.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
	# loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < 5:
            continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
#        find color of moving object
        bb = frame[y+h//4:y+3*h//4,x+w//4:x+3*w//4,:];
        color = np.mean(bb,axis=0)
        color = np.floor(np.mean(color, axis=0))
        # just select middle point intensity
#        color = np.array([frame[y+h//2,x+w//2,0], frame[y+h//2,x+w//2,1], frame[y+h//2,x+w//2,2]],dtype='float')
        cv2.rectangle(frame, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)    
#        cv2.rectangle(thresh, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)    

    cv2.imshow('thresh',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#    out.write(frame)
#out.release()
#%%
#cv2.rectangle(thresh, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)    
#
#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
cv2.destroyAllWindows()