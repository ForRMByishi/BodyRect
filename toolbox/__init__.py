import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import *
from imutils import perspective
from imutils import contours
import imutils
from unittest import case
from scipy.interpolate import spline
import scipy
import scipy.misc
import scipy.cluster
from scipy.stats import itemfreq
from sklearn import preprocessing

color_arr = [(204,0,204), (76,0,153), (0,0,102), (0,76,153), (0,204,204), (0,255,128), (0,204,0), (76,153,0), (204,204,0), (255,128,0), (204,0,0)]

def drawFocusRect(orig):
    h,w,c = orig.shape
    pad_left = int(0.15*w)
    pad_right = int(0.85*w)
    pad_bottom = int(0.8*h)
    pad_top = int(0.3*h)
    col = centralCol(orig,pad_left,pad_right,pad_top,pad_bottom)
    print("Left="+str(pad_left)+", Right="+str(pad_right)+",Width="+str(w)+",Height="+str(h)+", Color="+str(col))
    #mask = cv2.inRange(col2, lower_blue, upper_blue)
    #cv2.rectangle(orig, (pad_left,pad_top), (pad_right, pad_bottom), (int(col[0]),int(col[1]),int(col[2])), thickness=2, lineType=4)
    cv2.rectangle(orig, (pad_left,pad_top), (pad_right, pad_bottom), (255,255,255), thickness=2, lineType=4)
    return pad_left, pad_right, pad_top, pad_bottom,col

def inBound(pl,pr,pt,pb, c):
    penalty = 0
    result = False
    leftmost = tuple(c[c[:,:,0].argmin()][0])
    rightmost = tuple(c[c[:,:,0].argmax()][0])
    topmost = tuple(c[c[:,:,1].argmin()][0])
    bottommost = tuple(c[c[:,:,1].argmax()][0])
    if (pl < leftmost[0]) and (pr > rightmost[0]):
        result = True
        print("Leftmost="+str(leftmost)+"> pl="+str(pl)+", Rightmost ="+str(rightmost)+"< pr="+str(pr))
        if (pt < topmost[1]):
            penalty=+1
        if (pb< bottommost[1]):
            penalty=+1
    else:
        result = False
        penalty=2
    return result, penalty

def centralCol(orig, pl,pr,pt,pb):
    #returns the most abundant color inside the box specified by pl (padding left)...
    x_interval = int((pr-pl)/5)
    y_interval = int((pb-pt)/5)
    x1 = pl+(2*x_interval)
    x2 = pl+(3*x_interval)
    y1 = pt+(y_interval)
    y2 = pt+(2*y_interval)
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    Z = np.float32(img[y1:y2,x1:x2])
    pixels=Z.reshape((-1,3))
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    n_colors = 5
    _,labels,centers=cv2.kmeans(pixels,n_colors,None,criteria,10,flags)
    # Now convert back into uint8, and make original image
    palette = np.uint8(centers)
    #find the most abundant color in the kmeans cluster end return it in BGR format
    dominant_color = palette[np.argmax(itemfreq(labels)[:,-1])]
    return dominant_color



def returnCroppedSymmetricals(orig, refPt_left, refPt_right):
    #schneidet die halbierten Bildausschnitte anhand der Punktearrays aus und gibt sie zurück
    roi_l = orig[refPt_left[1]:refPt_left[3], refPt_left[0]:refPt_left[2]]
    roi_r = orig[refPt_right[1]:refPt_right[3], refPt_right[0]:refPt_right[2]]
    return roi_l, roi_r

def midpoint(ptA, ptB):
    #berechnet die Mitte aus zwei Punkten
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def checkWidthHeight(refPt_left, refPt_right):
    #überprüft Breite und Höhe aus dem Punktearray
    #refPt [0] ist topmost-left x
    #refPt [1] ist topmost-left y
    #refPt [2] ist bottommost-right x
    #refPt [3] ist bottommost-right y
    #korrigiert die Werte auf kürzere Abstände, wenn rechts und links unterschiedlich sein sollten
    #gibt die Höhe der Box zurück
    width_r = refPt_right[2]-refPt_right[0]
    width_l = refPt_left[2]-refPt_left[0]
    height_r = refPt_right[3]-refPt_right[1]
    height_l = refPt_left[3]-refPt_left[1]
    iw = width_l - width_r
    ih = height_l - height_r
    if (iw < 0):
        refPt_right[2]+=iw
        width_r+=iw
    elif (iw > 0):
        refPt_left[2]-=iw
        width_l-=iw
    if (ih < 0):
        refPt_right[3]+=ih
        height_r+=ih
    elif (ih > 0):
        refPt_left[3]-=ih
        height_l-=ih
    return height_r


def drawEdgedSymmetricals(orig, cnts, c):
    #malt eine halbierende Box um eine Kontur
    #gibt die Punktearrays für linke und rechte Hälfte zurück
    global color_arr
    #cv2.drawContours(orig, cnts, cnts.index(c), color_arr[cnts.index(c)], 2)
    leftmost = tuple(c[c[:,:,0].argmin()][0])
    rightmost = tuple(c[c[:,:,0].argmax()][0])
    topmost = tuple(c[c[:,:,1].argmin()][0])
    bottommost = tuple(c[c[:,:,1].argmax()][0])
    (midx, midy) = midpoint(leftmost, rightmost)
    cv2.line(orig,(int(midx),int(midy)),(int(midx),int(midy)+29),(255,0,0),3)
    vrefPt_left = [leftmost[0], topmost[1], int(midx), bottommost[1]]
    vrefPt_right = [int(midx), topmost[1], rightmost[0], bottommost[1]]
    cv2.rectangle(orig,(leftmost[0],topmost[1]),(int(midx),bottommost[1]), (255,0,0),1)
    cv2.rectangle(orig,(int(midx),topmost[1]),(rightmost[0],bottommost[1]), (255,128,0),1)
    return vrefPt_left, vrefPt_right

def returnSmoothParams(sub_g, height_r):
    #erzeugt eine Linie aus dem Bild (von oben nach unten)
    x = []
    y = []
    for index in range(height_r-1):
        pixSum= np.mean(sub_g[index]) #Grautonpixel auf einer y-Höhe zusammenaddieren
        x.append(int(index))
        y.append(int(pixSum))       
    x_sm = np.array(x)
    y_sm = np.array(y)
    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
    y_smooth = spline(x,y,x_smooth)
    return x_smooth, y_smooth

#normalize values
def scaleCoords(coords):
    minmaxScaler = preprocessing.MinMaxScaler()
    newCoords = np.copy(coords)
    i=0
    for c in coords:
        #Scaling the Coords, first X, then Y
        newCoords[i][0] = minmaxScaler.fit_transform(np.reshape(c[0],(-1,1))).flatten()
        newCoords[i][1] = minmaxScaler.fit_transform(np.reshape(c[1],(-1,1))).flatten()
        i+=1
    return newCoords

def readImgResize(imgname):
    # Read Image
    image = cv2.imread(imgname)
    resized = (imgResize(400,image))
    return resized

def imgResize(factor, image):
    # Resizing
    f = factor / image.shape[1]
    dim = (factor, int(image.shape[0] * f))
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

def crange(start,stop):
    r=int(stop-start)
    c=int(start+(r/2))
    iup=c
    idown=c
    lst=[None]*(r+1)
    i=0
    lst[i]=c
    while (iup<stop):
        i+=1
        iup+=1
        lst[i]=iup
        i+=1
        idown-=1
        lst[i]=idown
    return lst
        
    
