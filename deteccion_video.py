from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
#==============================================================================
import numpy as np
# # import cv2
from skimage import io
from skimage import morphology,data, filters, measure
from matplotlib import pyplot as plt

from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table

#%% Funciones

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img,scale):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    img=cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
    
    return img


def crackdetector(clase,img,x,y,w,h,d):
    x=np.abs(np.int(x))
    y=np.abs(np.int(y))
    w=np.int(w)
    h=np.int(h)
    
    img = img[y:y+h,x:x+w]
    
    if clase==0 or clase==1:
        
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        im = cv2.GaussianBlur(img_g,(3,3),0)

        #%%
        lut_1=np.arange(0,128,1.5,dtype='uint8')
        lut_2=255*np.ones((255-np.size(lut_1)+1),dtype='uint8')

        lut=np.concatenate([lut_1,lut_2])
        prueba=cv2.LUT(im,lut)

        #%%
        thresh = cv2.adaptiveThreshold(prueba,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY_INV,501,180)

        #%%
        arr = thresh > 0

        cleaned = morphology.remove_small_objects(arr, 30)
        cleaned = morphology.remove_small_holes(cleaned, 10)

        cleaned=np.uint8(255*cleaned)
        #%%
        kernel=np.ones((3,3),'uint8')

        dilat=cv2.dilate(cleaned,kernel,iterations=15)
        ero=cv2.erode(dilat,kernel,iterations=12)
        #%%
        result = img.copy()

        cnts, heirarchy = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        thr_size = 200
        grietas = [cv2.boundingRect(cnt) for cnt in cnts if cv2.contourArea(cnt) > thr_size]
        
        if len(grietas) > 0:
        
            d_h=np.zeros((1,len(grietas)))
            d_v=np.zeros((1,len(grietas)))

            for i, p in enumerate(grietas):
                x = p[0]
                y = max(0, p[1]-10)
                d_h[0,i] = p[2]
                d_v[0,i] = p[3]
                # cv2.putText(result, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            mask = np.zeros(ero.shape, np.uint8)
            cv2.drawContours(mask,cnts,-1,255, -1)
            mean,_,_,_ = cv2.mean(ero, mask=mask)

            cv2.drawContours(result, cnts, -1, (0,0,255), 1, lineType=cv2.LINE_AA)
                        
            if clase==0:
                ancho_grieta=d_h.max()
                print("Se detectó una grieta {} de {} ancho y {} largo".format(classes[int(clase)], ancho_grieta, h))
                filename = "imagenes/longitudinal/grieta_longitudinal_%d.jpg"%d
                cv2.imwrite(filename,result)
            elif clase==1:
                ancho_grieta=d_v.max()
                print("Se detectó una grieta {} de {} ancho y {} largo".format(classes[int(clase)], ancho_grieta, w))
                filename = "imagenes/transversal/grieta_transversal_%d.jpg"%d
                cv2.imwrite(filename,result)
        
        
    else:
        print("Se detectó una grieta {}".format(classes[int(clase)]))
        filename = "imagenes/cocodrilo/grieta_cocodrilo_%d.jpg"%d
        cv2.imwrite(filename,img)

    
        
    
    
#%%
d=0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (3840*2,2160*2))
    else: 
        cap = cv2.VideoCapture(opt.directorio_video)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (3840*2,2160*2))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (3840*2,2160*2), interpolation=cv2.INTER_CUBIC)
        #La imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        frame_c=frame.copy()
        
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    cv2.rectangle(frame_c, (x1, y1 + box_h), (x2, y1), color, 30)
                    cv2.putText(frame_c, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                    cv2.putText(frame_c, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    
                    crackdetector(cls_pred,frame,x1,y1,box_w,box_h,d)
                    d=d+1
                    
        #Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        
        if opt.webcam==1:
            frame_c=Convertir_BGR(frame_c,0.2)
            cv2.imshow('frame', frame_c)
            out.write(RGBimg)
        else:
            frame_c=Convertir_BGR(frame_c,0.2)
            out.write(frame_c)
            cv2.imshow('frame', frame_c)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    
