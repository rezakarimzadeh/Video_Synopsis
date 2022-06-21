import cv2
import numpy as np
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
#%% 
def remove_overlapping_bb(coordinates):
    removing_rows = []
    for k in range(len(coordinates)):
        (x1, y1, w1, h1)= coordinates[k]
        for l in range(len(coordinates)):
            if k != l: 
                (x2, y2, w2, h2)= coordinates[l]
                if x1< x2  and y1<y2 and (x1+w1)>(x2+w2) and (y1+h1)>(y2+h2):
                    removing_rows.append(l)
    if removing_rows:
        coordinates = np.delete(coordinates, removing_rows, 0)
    return coordinates

def centroid_calc(coordinates):
    centroid = []
    for j in range(len(coordinates)):
        (x, y, w, h)= coordinates[j]
        centroid.append([x+w//2, y+h//2])
    centroid = np.array(centroid)
    return centroid

def remove_corners_bb(coordinates):
    removing_rows = []
    for k in range(len(coordinates)):
        (x1, y1, w1, h1)= coordinates[k]
        if y1+h1//2<150:
            removing_rows.append(k)
        elif 480- y1 + x1 <300:
            removing_rows.append(k)
        elif (800-x1)/2 + 480- y1 <300:
            removing_rows.append(k)
    coordinates = np.delete(coordinates, removing_rows, 0)
    return coordinates  
background = cv2.imread('background1.jpg')
pMOG2 = cv2.createBackgroundSubtractorMOG2()
all_coordinates = []
all_centroid = []
bb_color = []
for i in range(len(seq)//1):
    frame2 = seq[i]
    frame = cv2.GaussianBlur(frame2, (9,9), 0)
    
    fgmaskMOG = pMOG2.apply(frame)
#    fgmaskMOG = cv2.absdiff(frame,background)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closing = cv2.morphologyEx(fgmaskMOG, cv2.MORPH_CLOSE, kernel)
    img_erosion = cv2.erode(closing, kernel, iterations=3)
    thresh = cv2.threshold(img_erosion, 240, 255, cv2.THRESH_BINARY)[1]
    

    #convexise
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_TC89_L1)
    contours = cnts[1]


    src = frame2.copy()
    coordinates=[]

    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < 50:
            continue  
        h =cv2.convexHull(contours[j])
        coordinates.append(cv2.boundingRect(h))
    coordinates = np.array(coordinates)
    coordinates = remove_overlapping_bb(coordinates)
    coordinates = remove_corners_bb(coordinates)
    all_coordinates.append(coordinates)
    
    #calculate centroides
    centroid = centroid_calc(coordinates)
    
    all_centroid.append(centroid)
    
    temp = []
    for j in range(len(coordinates)):
        (x, y, w, h)= coordinates[j]
        bb = src[y+h//4:y+3*h//4,x+w//4:x+3*w//4,:];
        color = np.mean(bb,axis=0)
        color = np.floor(np.mean(color, axis=0))
        temp.append(color)
    
        cv2.rectangle(src, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)
#        cv2.drawContours(src, hull, -1,(0,255,0),3)
    #    fgmaskMOG = cv2.threshold(, 200, 255, cv2.THRESH_BINARY)[1]
    bb_color.append(temp)
    cv2.imshow('frame',src)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
#%% convert lists to numpy array
all_centroid = np.array(all_centroid)
all_coordinates = np.array(all_coordinates)
bb_color = np.array(bb_color)

#%% 
import math

frame_vs_ids = []
idx_vs_coordinate = []
correspond_frame = []
idx = 0;
dist_thresh = 30
for i in range(len(seq)-1):
    if idx == 0:
        if all_coordinates[i].any():
            temp_co = []
            for j in range(len(all_coordinates[i])):
                idx += 1
                c = all_centroid[i][j]
                co = all_coordinates[i][j]
                color = bb_color[i][j]
                temp_co.append([idx,c,co,color])
            idx_vs_coordinate.append(temp_co)
            correspond_frame.append(i)
    else:
        last_fr = idx_vs_coordinate[-1]
        temp_co = []
        for j in range(len(all_centroid[i+1])):
            c1=all_centroid[i+1][j]
            closest_coo = all_coordinates[i+1][j]
            color = bb_color[i+1][j]
            mindist = 100
                       
            for idd,c2,_,_ in last_fr:
                dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist <mindist:
                    mindist = dist
                    closest_id = idd
                    closest_cent = c1
                    
            if mindist < dist_thresh:              
                temp_co.append([closest_id,closest_cent,closest_coo,color]) 
            else:
                idx=idx+1
                temp_co.append([idx,c1,closest_coo,color])
            
        idx_vs_coordinate.append(temp_co)
        correspond_frame.append(i+1)
                
#%%



id_org_coor_fr = {i:list() for i in range(1,idx+1)}
for i in range(len(correspond_frame)):
    c_f = correspond_frame[i]
    for car_id,org,coor,color in idx_vs_coordinate[i]: 
        id_org_coor_fr[car_id].append([org,coor,c_f,color])

#%% delete obects that exist < 6 frame
delete_idx = []
for i in range(1,len(id_org_coor_fr.keys())+1):
    if len(id_org_coor_fr[i])<6:
        delete_idx.append(i)
        del id_org_coor_fr[i]
#%% velocity
downToUp = 0
UpToDown = 0
car_id_velocity = {i:list() for i in id_org_coor_fr.keys()}
for i in id_org_coor_fr.keys():
    org1,_,c_f1,_ = id_org_coor_fr[i][0]
    org2,_,c_f2,_ = id_org_coor_fr[i][-1]
    velocity = int((org1[1]-org2[1])/(c_f2-c_f1)*50)
    car_id_velocity[i].append(velocity)
    if velocity > 0:
        downToUp +=1
    else:
        UpToDown +=1
print('Number of Cars go Up: ', downToUp)    
print('Number of Cars go Down: ', UpToDown)    
#%%
def GetMaxFlow(flows):        
    maks=max(flows, key=lambda k: len(flows[k]))
    return maks
frame_shape = seq[0].shape

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
text_color = (255, 255, 255) 
  
# Line thickness of 1 px 
thickness = 1
max_fr = GetMaxFlow(id_org_coor_fr)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('synopsisVid2.mp4',fourcc, 20, (800,480))

show_original_time = True 
show_car_color = False
show_velocity =  False

for i in range(max_fr):
    bacg = background.copy()
    for j in id_org_coor_fr.keys():
        mask1 = np.zeros(frame_shape,dtype='uint8')
        mask2 = np.ones(frame_shape,dtype='uint8')
        if i<len(id_org_coor_fr[j]) :
            org, [x,y,w,h], fr_num, color = id_org_coor_fr[j][i]
            frame = seq[fr_num]
            src = frame.copy()
            mask1[y:y+h,x:x+w,:] = 1
            src2 = src*mask1
            mask2[y:y+h,x:x+w,:] = 0
            bacg = bacg*mask2
            bacg = cv2.add(bacg,src2)
            if show_original_time:
                bacg = cv2.putText(bacg, str(int(fr_num/3)/10), (x,y), font, fontScale, text_color, thickness, cv2.LINE_AA) 
            if show_car_color:
                cv2.rectangle(bacg, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)
            if show_velocity:
                bacg = cv2.putText(bacg,str(car_id_velocity[j][0]) , (x,y), font, fontScale, text_color, thickness, cv2.LINE_AA)
    cv2.imshow('frame',bacg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    out.write(bacg)
out.release()
cv2.destroyAllWindows()      
            
#%% Gradient Line
gr_line = cv2.imread('background1.jpg')

for i in id_org_coor_fr.keys():
    start = True
    steps = len(id_org_coor_fr[i])
    brightness = 255//steps
    c = 0
    for org,_,_,_ in id_org_coor_fr[i]:
        c += brightness
        if start:
            org1 = org
            start = False
            continue
        org2 = org
        cv2.line(gr_line,(org1[0],org1[1]),(org2[0],org2[1]),(c,c,c),thickness = 3)
        org1 = org2
                    
cv2.imshow('Gradient Lines',gr_line)
k = cv2.waitKey(0) 
cv2.destroyAllWindows()     

                 
                
                

        
            






    