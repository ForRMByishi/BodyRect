import numpy as np
import cv2
import imutils
from scipy import ndimage
from numpy import float32
from toolbox import *


class BodyRect:
#     Image Variables:
#      - backup: input image that can be resetted
#      - image: input image as it is
#      - img_hsv: HSV Converted image
#      - mask: Skin-Color-Range mask
#      - masked: image masked through mask
#      - masked_grey: masked image converted ino grey (after RGB conversion)
#      - edged: after edge test_outlier_detection
#      - rect: Contour and Mask selected ROI of image
#      - maskedRect: Contour and Mask selected ROI of masked
#      - maskedGreyRect: maskedRect from Grey image
#      
#     Other Variables:
#      - quality: is white pixels/black pixels in the maskedRect-SkinColorMask
#      - rectShape: is (x,y,w,h) of the ROI in image
#      - center: contour-Center (x,y)
#      - heightBoundary: Dictionary (upperY and lowerY) initialized to 0. Will be set by user to mark upper and lower border for normalization
    
    # define range of skin color in HSV, following partly the paper of Kolkur et al.
    # Human Skin Detection Using RGB, HSV and YCbCr Color Models
    lower_skin_hsv = np.array([0,58,0])
    upper_skin_hsv = np.array([25,173,255])

    def __init__(self, img, autodetect='on'):
        self.quality = 0
        self.image=img
        if autodetect=='off':
            self.rect=img
            self.masked = img
            self.masked_grey = cv2.cvtColor(self.masked, cv2.COLOR_RGB2GRAY)
            self.maskedRect=self.masked
            self.maskedGreyRect = self.masked_grey
            rh, rw,_ = img.shape
            self.rectShape=(0,0,rw,rh)
            return
        
        self.backup=img.copy()
        self.img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
    
        # Threshold the HSV image to get only skin colors
        self.mask = cv2.inRange(self.img_hsv, BodyRect.lower_skin_hsv, BodyRect.upper_skin_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

        self.masked = cv2.bitwise_and(self.image,self.image, mask= self.mask)
        self.masked_rgb = cv2.cvtColor(self.masked, cv2.COLOR_HSV2RGB)
        self.masked_grey = cv2.cvtColor(self.masked_rgb, cv2.COLOR_RGB2GRAY)
    
        # Grey conversion and Canny edge detection of the cropped image
        
        self.edged = cv2.Canny(self.masked, 50, 100)
        self.edged = cv2.dilate(self.edged, kernel, iterations=1)
        self.edged = cv2.erode(self.edged, kernel, iterations=1)
        

        # now finding contours and combining them into one array
        cnts = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[1]
        #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # sort contours and keep only the largest
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        for c in cnts:
            rx, ry, rw, rh = cv2.boundingRect(c)
            #defining the center of the contour
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            self.center=(cx,cy)
            #now measuring the quantity of skin-colored contents in the rectangle = quality of the contour for our purposes
            mask2 = cv2.inRange(self.img_hsv[ry:ry+rh, rx:rx+rw], BodyRect.lower_skin_hsv, BodyRect.upper_skin_hsv)
            true_pixels = np.count_nonzero(mask2)
            false_pixels = mask2.size-true_pixels
            self.quality=true_pixels/false_pixels
            
        self.rect=self.image[ry:ry+rh, rx:rx+rw]
        self.maskedRect=self.masked[ry:ry+rh, rx:rx+rw]
        self.maskedGreyRect = self.masked_grey[ry:ry+rh, rx:rx+rw]
        self.rectShape=(rx,ry,rw,rh)
        self.heightBoundary={'upperY':0, 'lowerY':0}
        
    def retrieveRectDraw(self):
        img = self.rect.copy()
        h,w,_ = img.shape 
        cv2.line(img,(int(w/2), 0),(int(w/2),h),(0,0,255),1 )
        return img
        
    def retrieveSubR(self):
        img = self.maskedGreyRect.copy()
        h,w = img.shape 
        roi_l=img[0:h, 0:int(w/2)]
        roi_r=img[0:h, int(w/2):w]
        _,wl = roi_l.shape
        _,wr = roi_r.shape
        if (wl < wr):
            roi_r=img[0:h, int(w/2):(w-1)]
        elif (wl>wr):
            roi_l=img[0:h, 1:int(w/2)]

        sub_g = cv2.absdiff(roi_r, cv2.flip(roi_l, 1))
        
        return sub_g

    def retrieveSubL(self):
        img = self.maskedGreyRect.copy()
        h,w = img.shape 
        roi_l=img[0:h, 0:int(w/2)]
        roi_r=img[0:h, int(w/2):w]
        _,wl = roi_l.shape
        _,wr = roi_r.shape
        if (wl < wr):
            i=wr-wl
            roi_r=img[0:h, int(w/2):(w-i)]
        elif (wl>wr):
            i=wl-wr
            roi_l=img[0:h, i:int(w/2)]

        sub_g = cv2.absdiff(roi_l, cv2.flip(roi_r, 1))
        
        return sub_g
    
    def retrieveSubT(self):
        img = self.maskedRect.copy()
        roi_l=self.retrieveSubL()
        roi_r=self.retrieveSubR()
        h,wl,_=roi_l.shape
        h,wr,_=roi_r.shape
        _,w,_=img.shape
        img[0:h,0:wl]=roi_l
        img[0:h,(w-wr):w]=roi_r
        return img

    def imgWithSubInlet(self):
        rx,ry,_,_=self.rectShape
        img=self.image.copy()
        inlet=self.retrieveSubT()
        h,w,_=inlet.shape
        img[ry:ry+h, rx:rx+w]=inlet
        return img


    def rot(self, image, xy, angle):
        im_rot = ndimage.rotate(image,angle) 
        org_center = (np.array(image.shape[:2][::-1])-1)/2.
        rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        return im_rot, new+rot_center
    
    def optimizeRotShift(self):
        #optimizes Rotation and Shift of an image
        # Returns a tuple of the correct values (angle, x-Shift)
        x,y,w,h=self.rectShape
        rectPos=(x,y)
        orig=self.masked_grey.copy()
        #Angle and Shift dictionary to collect values
        as_Means={}
        for i in crange (-10,10):
            #Rotation optimization
            rotated, newXY=self.rot(orig, np.array(rectPos),i)
            for ii in crange(-20,20):
                #Shift optimization
                x=int(newXY[0])+ii
                y=int(newXY[1])
                self.maskedGreyRect=rotated[y:y+h,x:x+w]
                n=self.retrieveSubL() 
                sub_mean=np.mean(n, dtype=float32)
                as_Means.update({sub_mean:(i,ii)})
        #get the list with angle, shift coordinates of the lowest subtracted area
        a=min(as_Means.items(), key=lambda x: x[0])[1]
        #Setting self-values to the new rotation
        self.image,newXY=self.rot(self.image, np.array(rectPos),int(a[0]))
        self.masked,newXY=self.rot(self.masked, np.array(rectPos),int(a[0]))
        self.masked_grey,newXY=self.rot(self.masked_grey, np.array(rectPos),int(a[0]))
        #correct the x,y,center values and the inlets in the rotated image for the shift
        x=int(newXY[0])+a[1]
        y=int(newXY[1])
        self.rectShape=(x,y,w,h)
        self.rect=self.image[y:y+h,x:x+w]
        self.maskedRect=self.masked[y:y+h,x:x+w]
        self.maskedGreyRect=self.masked_grey[y:y+h,x:x+w]
        return a
    
    def retrieveSubQuarts(self):
        #returns subsets of the subtracted grey images along the x-axis starting in the middle
        # [0] being the first quart
        # [1] being the first half
        # [2] being 3/4
        # [3] being all
        sub=self.retrieveSubR()
        h,w=sub.shape
        w4=w/4
        q_arr=[]
        for i in range(1,5):
            ra=int(w4*i)
            q=sub[0:h,0:ra]
            q_arr.append(q)
        return q_arr    
    
    # mouse callback function
    def draw_circle(self,event,x,y,flags,param):
        _,cy = self.center
        if event == cv2.EVENT_MOUSEMOVE:
            img=self.img_anatomy.copy()
            if (y<cy):
                #if Mouse is above the center of the captured image
                cv2.line(img,(x,y),(x,y+15),(0,0,255),1)
                cv2.circle(img,(x,y),4,(0,0,255),1)
                cv2.putText(img,'Please indicate upper boundary',(x-30,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
                cv2.imshow('Anatomy', img)
            else:
                #if Mouse is below the center of the captured image
                cv2.line(img,(x,y),(x,y-15),(0,0,255),1)
                cv2.circle(img,(x,y),4,(0,0,255),1)
                cv2.putText(img,'Please indicate lower boundary',(x-30,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
                cv2.imshow('Anatomy', img)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img_anatomy,(x,y),4,(255,0,255),-1)
            if (y<cy):
                self.heightBoundary['upperY']=y
                cv2.putText(self.img_anatomy,'Upper boundary',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
            else:
                self.heightBoundary['lowerY']=y
                cv2.putText(self.img_anatomy,'Lower boundary',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
            cv2.imshow("Anatomy",self.img_anatomy)
    
    def getNormalizedXY(self, sub,y2,y1):
        minmaxScaler = preprocessing.MinMaxScaler()
        x = []
        y = []
        for index in range((y2-y1)-1):
            pixSum= np.mean(sub[index]) #Grautonpixel auf einer y-Höhe zusammenaddieren
            x.append(int(index))
            y.append(int(pixSum))       
        x_sm = np.array(x)
        y_sm = np.array(y)
        x_sm2 = minmaxScaler.fit_transform(np.reshape(x_sm,(-1,1))).flatten()
        y_sm2 = minmaxScaler.fit_transform(np.reshape(y_sm,(-1,1))).flatten()
        x_smooth = np.linspace(x_sm2.min(), x_sm2.max(), 200)
        y_smooth = spline(x_sm2,y_sm2,x_smooth)
        return x_smooth, y_smooth

    
    def returnNormalizedXY(self, img, autodetect = False):
        #returns a normalized image array of the subtracted R image
        
        self.img_anatomy=self.maskedRect.copy()
        if self.heightBoundary['lowerY'] == 0:
            if autodetect == False:
                cv2.imshow('Anatomy',self.img_anatomy)
                cv2.setMouseCallback('Anatomy',self.draw_circle)
                while(1):
                    k = cv2.waitKey(1) & 0xFF
                    if ((k == ord('a')) and (self.heightBoundary['lowerY'] > 0) and (self.heightBoundary['upperY'] > 0)):  
                        y1=int(self.heightBoundary['upperY'])
                        y2=int(self.heightBoundary['lowerY'])       
                        _,w=img.shape
                        sub=img.copy()[y1:y2,0:w]
                        return self.getNormalizedXY(sub, y2, y1)
                    if (k== ord('d')):
                        return [],[]
        else:
            y1=int(self.heightBoundary['upperY'])
            y2=int(self.heightBoundary['lowerY'])       
            _,w=img.shape
            sub=img.copy()[y1:y2,0:w]
            return self.getNormalizedXY(sub, y2, y1) 
                
                
            
 