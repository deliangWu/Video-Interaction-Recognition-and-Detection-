'''A file for person detection (HOG + Linear SVM) and tracking (Kalman Filter)'''
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import numpy as np
import cv2
import videoPreProcess as vp

'''video normalization '''
def imgNorm(imgIn):
    vmax = np.amax(imgIn)
    vmin = np.amin(imgIn)
    vo = np.uint8((imgIn - vmin)/(vmax-vmin) * 255)
    return vo

'''HOG + Linar SVM person detector'''
def humanDetector(video, dispBBEn = False):
    # initalize the HOG descriptor and person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    i = 0
    picks = []
    for frame in video:
        img = imutils.resize(frame,height = min(200,frame.shape[1]))
        # detect the people in the image
        (rects,weights) = hog.detectMultiScale(img,winStride=(2,2),padding=(16,16),scale=1.05)
        # draw the original bounding boxes
        rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
        pick = non_max_suppression(rects,probs=None,overlapThresh=0.65)
        if dispBBEn is True: 
            for (xA,yA,xB,yB) in pick:
                cv2.rectangle(img,(xA,yA),(xB,yB),(0,255,0),1)
       
        img = np.reshape(img,(1,) + img.shape)
        picks.append(pick)
        if i == 0 :
            vo = img.copy()
        else:
            vo = np.append(vo,img,axis = 0)
        i+= 1        
    return (vo,picks)

'''Calculate the Euclidian distance between two points'''
def distCalc(p1,p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

'''Reshape bounding boxes'''
def bbReShape(boundingBoxes,width,height,frmSize):
    def move(a,b):
        if a - b < 0:
            o = 0
        else:
            o = a - b
        return o
    def mux(cond,a,b):
        if cond is True:
            f = a
        else:
            f = b
        return f
    def normBB(pos_lt,width,height,frmSize):
        if (pos_lt[0] + width > frmSize[1]):
            xA = int(frmSize[1] - width)
            xB = frmSize[1]
        elif pos_lt[0] < 0:
            xA = 0
            xB = width
        else:
            xA = pos_lt[0]
            xB = int(pos_lt[0] + width)
            
        if pos_lt[1] + height > frmSize[0]:
            yA = int(frmSize[0] - height)
            yB = frmSize[0]
        elif pos_lt[1] < 0:
            yA = 0
            yB = height
        else:
            yA = pos_lt[1]
            yB = int(pos_lt[1] + height)
        return ([xA, yA, xB, yB])
    
    pos_lts = [ np.array([bb[0][0],bb[0][1]],np.float32) for bb in boundingBoxes]    
    if pos_lts != []:
        if len(pos_lts) == 1:
            pos_lt_filtered = kalmanFilter(pos_lts[0],kalman0)
            bb_ori = (normBB([int(pos_lt_filtered[0]),int(pos_lt_filtered[1])], width, height, frmSize), boundingBoxes[0][1])
            bb_reshape = (normBB([move(int(pos_lt_filtered[0]),20),move(int(pos_lt_filtered[1]),0)], int(width * 1.3), height, frmSize), boundingBoxes[0][1])
            
            bb_ori = [bb_ori]
            bb_reshape = [bb_reshape]
        else:
            pos_lt_filtered = kalmanFilter(pos_lts[0],kalman0)
            bb_ori_0 = (normBB([int(pos_lt_filtered[0]),int(pos_lt_filtered[1])], width, height, frmSize), boundingBoxes[0][1])
            bb_reshape_0 = (normBB([move(int(pos_lt_filtered[0]),20),move(int(pos_lt_filtered[1]),0)], int(width * 1.3), height, frmSize), boundingBoxes[0][1])
            
            pos_lt_filtered = kalmanFilter(pos_lts[1],kalman1)
            bb_ori_1 = (normBB([int(pos_lt_filtered[0]),int(pos_lt_filtered[1])], width, height, frmSize), boundingBoxes[1][1])
            bb_reshape_1 = (normBB([move(int(pos_lt_filtered[0]),20),move(int(pos_lt_filtered[1]),0)], int(width * 1.3), height, frmSize), boundingBoxes[1][1])
            
            bb_ori = [bb_ori_0, bb_ori_1]
            bb_reshape = [bb_reshape_0, bb_reshape_1]
    else:
        bb_ori = []
        bb_reshape = []
    return ((bb_ori,bb_reshape))

'''Kalman filter'''
def kalmanFilter(point,kalman):
    kalman.correct(point)
    pred = kalman.predict()
    return pred

'''Segment a video according to the bounding box'''
def frameSegment(imageIn,boundingBox,frmSize):
    xA,yA,xB,yB = boundingBox
    img_crop = imageIn[yA:yB,xA:xB]
    #imgOut = np.reshape(np.array([119,136,153] * frmSize[0] * frmSize[1],dtype=np.uint8), frmSize + (3,))
    #resizeRatio = min([a/b for a,b in zip(frmSize,img_crop.shape[0:2])])
    #resizedImg = cv2.resize(img_crop,tuple([int(i * resizeRatio / 2) * 2 for i in reversed(img_crop.shape[0:2])]),interpolation=cv2.INTER_AREA)
    #x1,y1 = resizedImg.shape[0:2]
    #x2,y2 = frmSize
    #imgOut[int(abs(x2 - x1)/2) : int((x2 + x1)/2), int(abs(y2 - y1)/2) : int((y2 + y1)/2)] = resizedImg
    imgOut = cv2.resize(img_crop,(frmSize[1],frmSize[0]),interpolation=cv2.INTER_AREA)
    
    return imgOut

'''update a bounding boxes according to the results of person tracking'''
def updateBB(position_pre, pick, updateNo, updateFrameNo, currFrameNo,dists):
    print('position_pre ------------', position_pre)
    print('pick---------------------', pick)
    print('updateNo-----------------', updateNo)
    print('updateFrameNo------------', updateFrameNo)
    print('currentFrameNo-----------', currFrameNo)
    print('dists--------------------', dists)
    print('**************************************************************************')
    print('**************************************************************************')
    thld = 50
    if updateNo == 1:
        if abs(position_pre[0][0][0] - pick[0][0]) > (currFrameNo - updateFrameNo[0]) * thld:
            position_pre = position_pre
        else:
            position_pre= [(pick[0],0),position_pre[1]]
    elif updateNo == 2:
        if abs(position_pre[1][0][0] - pick[0][0]) > (currFrameNo - updateFrameNo[1]) * thld:
            position_pre = position_pre
        else:
            position_pre = [position_pre[0], (pick[0],1)]
    else:
        if abs(position_pre[0][0][0] - pick[np.argmin(dists)][0]) > (currFrameNo - updateFrameNo[0]) * thld and abs(position_pre[1][0][0] - pick[np.argmax(dists)][0]) > (currFrameNo - updateFrameNo[0]) * thld:
            position_pre = position_pre
        elif abs(position_pre[0][0][0] - pick[np.argmin(dists)][0]) < (currFrameNo - updateFrameNo[0]) * thld and abs(position_pre[1][0][0] - pick[np.argmax(dists)][0]) > (currFrameNo - updateFrameNo[0]) * thld:
            position_pre = [(pick[np.argmin(dists)],0), position_pre[1]]
        elif abs(position_pre[0][0][0] - pick[np.argmin(dists)][0]) > (currFrameNo - updateFrameNo[0]) * thld and abs(position_pre[1][0][0] - pick[np.argmax(dists)][0]) < (currFrameNo - updateFrameNo[0]) * thld:
            position_pre = [position_pre[0], (pick[np.argmax(dists)],1)]
        else:
            position_pre = [(pick[np.argmin(dists)],0), (pick[np.argmax(dists)],1)]
    return position_pre

def checkLabel(position_pre):
    if (position_pre[0][0][0] > position_pre[1][0][0]):
        position_pre = [(position_pre[1][0],0),(position_pre[0][0],1)]
    return position_pre
        
'''tracking two people with Kalman filtes'''    
def humanTracking(vIn,picks,frmSize = (112,80,3),dispBBEn = True):
    # initialize kalman filter
    global kalman0,kalman1
    kalman0 = cv2.KalmanFilter(4,2)
    kalman0.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman0.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman0.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1e-6 
    kalman0.measurementNoiseCov = np.identity(2,np.float32) * 1 
    kalman1 = cv2.KalmanFilter(4,2)
    kalman1.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman1.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman1.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1e-6
    kalman1.measurementNoiseCov = np.identity(2,np.float32) * 1 
    
    colors = [(0,255,0),(0,255,255)]
    # calculate the average width and height of a person
    widths = [pos[0][2] - pos[0][0] for pos in picks[500:-200] if pos != []]
    heights = [pos[0][3] - pos[0][1] for pos in picks[500:-200] if pos != []]
    width = int(sum(widths) / max(1,len(widths)))
    height = int(sum(heights) / max(1,len(heights)))
    # the format of position_pre for two bounding boxes is [([xA,yA,xB,yB],label), ([xA,yA,xB,yB],label)]    
    position_pre = []
    disp_bb_en_cnt = 0
    vOut_0 = vOut_1 = []
    bb_out = []
    for i in range(vIn.shape[0]):
        image = vIn[i]
        pick = picks[i]
        
        if position_pre == []:
            position_pre = [(pick,l) for l,pick in enumerate(pick[0:2])]
        elif len(position_pre) == 1:
            dists = [distCalc((xA,yA),(position_pre[0][0][0],position_pre[0][0][1])) for (xA,yA,xB,yB) in pick]
            if len(dists) == 0:
                position_pre = position_pre
            elif len(dists) == 1:
                if dists[0] > 100:
                    position_pre = [position_pre[0],(pick[0],1)]
                else:
                    position_pre = [(pick[0],0)]
            else:
                position_pre = [(pick[np.argmin(dists)],0),([pick[j] for j in range(len(dists)) if j != np.argmin(dists)][0],1)]
        else:
            position_pre = checkLabel(position_pre)
            dists0 = [distCalc((xA,yA),(position_pre[0][0][0],position_pre[0][0][1])) for (xA,yA,xB,yB) in pick]
            dists1 = [distCalc((xA,yA),(position_pre[1][0][0],position_pre[1][0][1])) for (xA,yA,xB,yB) in pick]
            if len(pick) == 0:
                position_pre = position_pre
            elif len(pick) == 1:
                if dists0 < dists1:
                    position_pre[0] = (pick[0],0)
                else:
                    position_pre[1] = (pick[0],1)
            else:
                position_pre[0] = (pick[np.argmin(dists0)],0)
                if np.argmin(dists1) == np.argmin(dists0):
                    index1 = [x for x in range(len(pick)) if x != np.argmin(dists0)][0]
                else:
                    index1 = np.argmin(dists1)
                position_pre[1] = (pick[index1],1)
        
        bb_ori,bb_reshape = bbReShape(position_pre,width,height,image.shape[0:2])
        
        if (len(bb_reshape) >= 2):
            disp_bb_en_cnt += 1
        # crop the video into two separated video volumes which contain one person for each.         
        if disp_bb_en_cnt == 20:
            bbInitFrmNo = i
        if disp_bb_en_cnt >= 20:
            bb_out.append(bb_reshape)
            cropped_images = [frameSegment(image,bb[0],frmSize=frmSize) for bb in bb_reshape]
            
            if vOut_0 == []:
                vOut_0 = np.reshape(cropped_images[0],(1,) + frmSize)
            else:
                vOut_0 = np.append(vOut_0, np.reshape(cropped_images[0],(1,) + frmSize),axis = 0)
            
            if vOut_1 == []:
                vOut_1 = np.reshape(cropped_images[1],(1,) + frmSize)
            else:
                vOut_1 = np.append(vOut_1, np.reshape(cropped_images[1],(1,) + frmSize),axis = 0)
            if dispBBEn == True: 
                for bb in bb_reshape:
                    cv2.rectangle(image,(bb[0][0],bb[0][1]),(bb[0][2],bb[0][3]),colors[bb[1]],1)
            
    return (vIn, vOut_0, vOut_1,bb_out,bbInitFrmNo)    