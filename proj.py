# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:01:54 2019

@author: Nikola
"""

from __future__ import print_function
#import potrebnih biblioteka
#%matplotlib inline
import cv2
#import os
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import model_from_json
#import collections

# keras
#from keras.models import Sequential
#from keras.layers.core import Dense,Activation
#from keras.optimizers import SGD

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova,zak ako nije potrebno


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 222, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    alphabet = [0,1,2,3,4,5,6,7,8,9]
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1'''   
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
pomX=[]
pomY=[]
mat =[]
dodato = 5
def select_roi(image_orig, image_bin, X1, Y1, X2, Y2, l):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    if (Y2-Y1)!=0 and (X2-X1)!=0:
        k1 = (Y2-Y1)/(X2-X1)
    else:
        k1 = 112.23  
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    
    
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        xy=[x,y]
          

        if (y-Y1)!=0 and (x-X1)!=0:
            k2 = (y-Y1)/(x-X1)
        else:
            k2 = 112.23   
  
        K = abs(k1-k2)
        v = abs((X1*Y2 + X2*y + x*Y1)-(x*Y2 + y*X1 + X2*Y1))
        #abs((X1*(Y2-(y-5))+X2*((y-5)-Y1)+(x-5)*(Y1-Y2))/2)
        bb= abs((X1*(Y2-(y-5))+X2*((y-5)-Y1)+(x-5)*(Y1-Y2))/2)
        #print(bb)# and x > X1-5 and x < X2+17
        if (bb<190)and (x > (X1-5)) and (x < (X2+17)) and h < 60 and h > 13 and w > 3:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom            
            l+=1
#            print(l)
#            print(v)
#            print(bb)
#            print(K)

            var = False
            for p in mat:
                if (p == xy):
                    var = True
            if var == False:
                mat.append(xy)                
                if (len(mat)) > 1:
                    uzastop = mat[-2]
            #print(x,y)
#             or (((uzastop[0])==xy[0]) or ((uzastop[0])==xy[0]) or ((uzastop[1])==xy[1]) or ((uzastop[1])==xy[1]))==True
            if var != True:
                if (len(mat)) > 1:                  
                    if not((((uzastop[0] - 1)==xy[0]) or ((uzastop[0] + 1)==xy[0]) or ((uzastop[1] - 1)==xy[1]) or ((uzastop[1] + 1)==xy[1])
                    or ((uzastop[0] - 2)==xy[0]) or ((uzastop[0] + 2)==xy[0]) or ((uzastop[1] - 2)==xy[1]) or ((uzastop[1] + 2)==xy[1]))):                                      
                        print("dobrooo", x)
                        print(K,bb, v)
#                        print(x,y)
                        region = image_bin[y:y+h+1,x:x+w+1]
                        regions_array.append([resize_region(region), (x,y,w,h)])       
                        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
                else:
                    print("jednomm")
                    #print(x,y,v)
                    region = image_bin[y:y+h+1,x:x+w+1]
                    regions_array.append([resize_region(region), (x,y,w,h)])       
                    cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
        
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]  
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, x, y


file= open("out.txt","w+")
file.write("RA 1/2015 Nikola Slijepcevic\r")
file.write("file	sum\r")

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
ann = model_from_json(model_json)
ann.load_weights("model.h5")

######
######
#listaKoordinataX = []
#listaKoordinataY = []
#cap = cv2.VideoCapture("video-8.avi") #video_name is the video being called
#cap.set(1,5); # Where frame_no is the frame you want
#ret, frame1 = cap.read() # Read the frame
##cv2.imshow('Frejm', frame1) # show frame on wind
#
##slika_gray = image_gray(frame1)
##slika_bin = image_bin(slika_gray)
##ret, slika_bin = cv2.threshold(slika_gray, 43, 255, cv2.THRESH_BINARY) # 78 granica prikazivanja, ret je vrednost praga, image_bin je binarna slika
#
#kernel = np.ones((3, 3))
#
#pom=frame1[:,:,0]
##cv2.imshow('Frejm1', pom) 
##erodirana = erode(slika_bin)
#pom1 = erode(pom)
##display_image(pom1)
##plt.grid(True)
#
#hough = cv2.HoughLinesP(pom1,1,np.pi/180,60,50,50)
#
#print(hough)
#row, c1, column = np.shape(hough)
#pomX1 = []
#pomY1 = []
#pomX2 = []
#pomY2 = []
#
#a = 1
#n = 0 # vrste
#while a < row+1:
#    pom1 = hough[:][n][0][0]
#    pom11 = hough[:][n][0][1]
#    pom2 = hough[:][n][0][2]
#    pom22 = hough[:][n][0][3]
#    
#    pomX1.append(pom1)
#    pomY1.append(pom11)
#    pomX2.append(pom2)
#    pomY2.append(pom22)
#    
#    n += 1
#    a += 1
#
#X1 = min(pomX1)
#Y1 = max(pomY1)
#X2 = max(pomX2)
#Y2 = min(pomY2)
#
#print(X1, Y1, X2, Y2)
####
####

#cap = cv2.VideoCapture('video-8.avi')
#if (cap.isOpened()== False): 
#  print("Error opening video stream or file")
  
l=0
broj_piksela=[0]
broj=0
for i in range(0,10):
    regioni_zbir = []
    cap = cv2.VideoCapture('video/video-'+str(i)+'.avi')
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        break
        ######
    ######
    listaKoordinataX = []
    listaKoordinataY = []
    #cap = cv2.VideoCapture("video-8.avi") #video_name is the video being called
    cap.set(1,5); # Where frame_no is the frame you want
    cap.read() # Read the frame
    ret, frame1 = cap.read()
    #cv2.imshow('Frejm', frame1) # show frame on wind
    
    #slika_gray = image_gray(frame1)
    #slika_bin = image_bin(slika_gray)
    #ret, slika_bin = cv2.threshold(slika_gray, 43, 255, cv2.THRESH_BINARY) # 78 granica prikazivanja, ret je vrednost praga, image_bin je binarna slika
    
    kernel = np.ones((3, 3))
    
    pom=frame1[:,:,0]
    #cv2.imshow('Frejm1', pom) 
    #erodirana = erode(slika_bin)
    pom1 = erode(pom)
    #display_image(pom1)
    #plt.grid(True)
    
    hough = cv2.HoughLinesP(pom1,1,np.pi/180,60,50,50)
    
    print(hough)
    row, c1, column = np.shape(hough)
    pomX1 = []
    pomY1 = []
    pomX2 = []
    pomY2 = []
    
    a = 1
    n = 0 # vrste
    while a < row+1:
        pom1 = hough[:][n][0][0]
        pom11 = hough[:][n][0][1]
        pom2 = hough[:][n][0][2]
        pom22 = hough[:][n][0][3]
        
        pomX1.append(pom1)
        pomY1.append(pom11)
        pomX2.append(pom2)
        pomY2.append(pom22)
        
        n += 1
        a += 1
    
    X1 = min(pomX1)
    Y1 = max(pomY1)
    X2 = max(pomX2)
    Y2 = min(pomY2)
    
    print(X1, Y1, X2, Y2)
    ####
    ####
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Display the resulting frame
        slika_gray = image_gray(frame)
        binarnaSlika = image_bin(slika_gray)
        
        image_orig, sorted_regions, x, o = select_roi(frame, binarnaSlika, X1, Y1, X2, Y2, l)
        cv2.imshow('Frame',image_orig) 
        for region in sorted_regions:
            regioni_zbir.append(region);
#            display_image(region)
#            plt.figure()
#            display_image(image_orig)
#            plt.figure()
           
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('c'):
          break
      # Break the loop
      else: 
        break 
    # When everything done, release the video capture object
    
    cap.release() 
    cv2.destroyAllWindows()
    rez = ann.predict(np.array(prepare_for_ann(regioni_zbir),np.float32))
    zbir11 = sum(display_result(rez));
    print(zbir11)
    file.write('video-'+str(i)+'.avi\t' + str(zbir11)+'\r')
    
file.close()

