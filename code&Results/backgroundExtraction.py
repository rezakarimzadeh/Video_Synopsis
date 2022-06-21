# extract background and track objects
import cv2
import numpy as np
cap = cv2.VideoCapture('Video1.avi')

seq=[]

while True:
	ret_val, frame = cap.read()
	if frame is None:
		break
	seq.append(frame)

# convert to numpy array
seq = np.array(seq)

# calculate median of arrays	
background = np.median(seq, axis=0).astype('uint8')
print('background shape: ',background.shape)
cv2.imwrite('background2.jpg',background)
cap.release()
cv2.imshow('background',background)
cv2.waitKey(0)
cv2.destroyAllWindows()










