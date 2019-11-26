import time
from pynq import Overlay
import numpy as np
from pynq import Xlnk
import struct
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt

def RunConv(conv,relu_en,sigmoid_en,feature_in,W,bias,feature_out):
    conv.write(0x10,relu_en);
    conv.write(0x18,sigmoid_en)
    conv.write(0x20,feature_in.physical_address);
    conv.write(0x28,W.physical_address);
    conv.write(0x30,bias.physical_address);
    conv.write(0x38,feature_out.physical_address);
    conv.write(0, (conv.read(0)&0x80)|0x01 );
    tp=conv.read(0)
    while not ((tp>>1)&0x1):
        tp=conv.read(0);
    #print(tp);
    
def Fusion(fusion,img1,img2,decision,fus):
    fusion.write(0x10,img1.physical_address);
    fusion.write(0x18,img2.physical_address);
    fusion.write(0x20,decision.physical_address);
    fusion.write(0x28,fus.physical_address);
    fusion.write(0, (conv.read(0)&0x80)|0x01 );
    tp=fusion.read(0)
    while not ((tp>>1)&0x1):
        tp=fusion.read(0);
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmid_all(x,h,w):

    s = np.zeros((h,w))
    for hh in range(h):
        for ww in range(w):
            s[hh][ww] = 1 / (1 + np.exp(-x[hh][ww]))
    return s
ol=Overlay("final.bit")
ol.ip_dict
ol.download()
conv=ol.Conv_half1_0
fusion = ol.fusion_0
ol.ip_dict['axi_gpio_0']
# ol?
from pynq.lib import AxiGPIO

button = ol.ip_dict['axi_gpio_0']
led =ol.ip_dict['axi_gpio_1']
switch = ol.ip_dict['axi_gpio_2']
button1 = AxiGPIO(button).channel1
led1 = AxiGPIO(led).channel1
switch1 = AxiGPIO(switch).channel1
#Conv1
IN_WIDTH=32
IN_HEIGHT=32
IN_CH=2

KERNEL_WIDTH=3
KERNEL_HEIGHT=3
X_STRIDE=1
Y_STRIDE=1

RELU_EN=1

if(MODE):
    X_PADDING=int((KERNEL_WIDTH-1)/2)
    Y_PADDING=int((KERNEL_HEIGHT-1)/2)
else:
    X_PADDING=0
    Y_PADDING=0

OUT_CH=32
OUT_WIDTH=int((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1)
OUT_HEIGHT=int((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1)
OUT = 1
xlnk=Xlnk();

ol=Overlay("final.bit")
ol.ip_dict
ol.download()
conv=ol.Conv_half1_0
fusion = ol.fusion_0
print("Overlay download finish");

#input image
image=xlnk.cma_array(shape=(IN_HEIGHT,IN_WIDTH,32),cacheable=0,dtype=np.float16)
# print(type(image))
image1 = xlnk.cma_array(shape=(IN_HEIGHT,IN_WIDTH,3),cacheable=0,dtype=np.float16)
# print(type(image1))
image2 = xlnk.cma_array(shape=(IN_HEIGHT,IN_WIDTH,3),cacheable=0,dtype=np.float16)
decision = xlnk.cma_array(shape=(IN_HEIGHT,IN_WIDTH),cacheable=0,dtype=np.float16)
#conv1
W_conv1=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,32,OUT_CH),cacheable=0,dtype=np.float16)
b_conv1=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv1=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv2=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv2=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv2=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv3=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv3=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv3=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv4=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv4=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv4=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv5=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv5=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv5=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv6=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv6=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv6=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv7=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv7=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv7=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv8=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv8=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv8=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv9=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH),cacheable=0,dtype=np.float16)
b_conv9=xlnk.cma_array(shape=(OUT_CH),cacheable=0,dtype=np.float16)
h_conv9=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,OUT_CH),cacheable=0,dtype=np.float16)

W_conv10=xlnk.cma_array(shape=(KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,32),cacheable=0,dtype=np.float16)
b_conv10=xlnk.cma_array(shape=(32),cacheable=0,dtype=np.float16)
h_conv10=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,32),cacheable=0,dtype=np.float16)
fus=xlnk.cma_array(shape=(OUT_HEIGHT,OUT_WIDTH,3),cacheable=0,dtype=np.float16)

#Initialize W, bias

w_conv1=np.load('./weights_npy/conv1_weights.npy')
W_conv1[:,:,0:2,:]=w_conv1
B_conv1=np.load('./weights_npy/conv1_bias.npy')

b_conv1[:]=B_conv1

w_conv2=np.load('./weights_npy/conv2_weights.npy')
W_conv2[:,:,:,:]=w_conv2
B_conv2=np.load('./weights_npy/conv2_bias.npy')
b_conv2[:]=B_conv2
w_conv2=np.load('./weights_npy/conv2_weights.npy')
W_conv2[:,:,:,:]=w_conv2
B_conv2=np.load('./weights_npy/conv2_bias.npy')
b_conv2[:]=B_conv2[:]

w_conv3=np.load('./weights_npy/conv3_weights.npy')
W_conv3[:,:,:,:]=w_conv3
B_conv3=np.load('./weights_npy/conv3_bias.npy')
b_conv3[:]=B_conv3

w_conv4=np.load('./weights_npy/conv4_weights.npy')
W_conv4[:,:,:,:]=w_conv4
B_conv4=np.load('./weights_npy/conv4_bias.npy')
b_conv4[:]=B_conv4
    
w_conv5=np.load('./weights_npy/conv5_weights.npy')
W_conv5[:,:,:,:]=w_conv5
B_conv5=np.load('./weights_npy/conv5_bias.npy')
b_conv5[:]=B_conv5
    
w_conv6=np.load('./weights_npy/conv6_weights.npy')
W_conv6[:,:,:,:]=w_conv6
B_conv6=np.load('./weights_npy/conv6_bias.npy')
b_conv6[:]=B_conv6
    
w_conv7=np.load('./weights_npy/conv7_weights.npy')
W_conv7[:,:,:,:]=w_conv7
B_conv7=np.load('./weights_npy/conv7_bias.npy')
b_conv7[:]=B_conv7
    
w_conv8=np.load('./weights_npy/conv8_weights.npy')
W_conv8[:,:,:,:]=w_conv8
B_conv8=np.load('./weights_npy/conv8_bias.npy')
b_conv8[:]=B_conv8
    
w_conv9=np.load('./weights_npy/conv9_weights.npy')
W_conv9[:,:,:,:]=w_conv9
B_conv9=np.load('./weights_npy/conv9_bias.npy')
b_conv9[:]=B_conv9

w_conv10=np.load('./weights_npy/conv10_weights.npy')
W_conv10[:,:,:,0:1]=w_conv10
B_conv10=np.load('./weights_npy/conv10_bias.npy')
b_conv10[0]=B_conv10


print("Finish initial")


import os
path = '/home/xilinx/jupyter_notebooks/project_final/picture_in'
# path = '/home/xilinx/jupyter_notebooks/project_final/paizhao_out'
# while 1:
#     if button1.read() == 1:
#         out_path = './paizhao_out'
#         output = out_path+ '/' +str(1)+'.jpg'
#         !fswebcam -F 30 --no-banner --save {output} -d /dev/video0 --resolution 640,480
#         break
# while 1:    
#     if button1.read() == 2:
#         out_path = './paizhao_out'
#         output = out_path+ '/' +str(2)+'.jpg'
#         !fswebcam -F 30 --no-banner --save {output} -d /dev/video0 --resolution 640,480    
#         break
        
dir = os.listdir(path)
while 1:
    if button1.read() == 1:
        for i in range(int(len(dir)/2)):
#         for i in range(2):  
            name1 = str(str(i+1)+'-A.jpg')
            print(name1)
            name2 = str(str(i+1)+'-B.jpg')
            print(name2)
            name_out = str(str(i+1)+'.jpg')

            img_color1 = cv2.imread(path+'/'+name1).astype(np.float32)
            img_color2 = cv2.imread(path+'/'+name2).astype(np.float32)
            print('we work on ',path+'/'+name1)

            img1=cv2.imread(path+'/'+name1,cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img2=cv2.imread(path+'/'+name2,cv2.IMREAD_GRAYSCALE).astype(np.float32)
            print("Read image")
            #image1=image1.reshape((IN_HEIGHT1,IN_WIDTH1,IN_CH1))
            inputsize = 32
            imgsize = 256   

            img_color1=cv2.resize(img_color1,(imgsize,imgsize)) 
            img_color2=cv2.resize(img_color2,(imgsize,imgsize))
            img1=cv2.resize(img1,(imgsize,imgsize)) 
            img2=cv2.resize(img2,(imgsize,imgsize))

            img_final = np.zeros((256,256,3))

            t6 =time.time()

            for h in range(int(imgsize/inputsize)):
                for w in range(int(imgsize/inputsize)):
#                     if i<16:
#                         led1[0:4].write(i)
#                         led1.off
            #         t0 =time.time()
                    img_color1_crop= img_color1[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1),:]
                    img_color2_crop= img_color2[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1),:]
                    image1[:,:,0:3] = img_color1_crop
                    image2[:,:,0:3] = img_color2_crop
                    img1_crop = img1[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)]
                    img2_crop = img2[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)]

            #         print(type(image))
            #         imageyy = np.zeros((32,32,32))
            #         print(inputsize*h,inputsize*(h+1))
            #         print(inputsize*w,inputsize*(w+1))
            #         t1 =time.time()
                    image[:,:,0] = img1_crop/255.0
                    image[:,:,1] = img2_crop/255.0
            #         print(type(image))
            #         for i in range(inputsize):
            #             for j in range(inputsize):
            #                 imageyy[i][j][0]=img1_crop[i][j]/255.0
            #         for i in range(inputsize):
            #             for j in range(inputsize):
            #                 imageyy[i][j][1]=img2_crop[i][j]/255.0
            #             print("Finish reading image")
                    t2 = time.time()
            #         print(t2-t1)
            #         print(w,h)
                    RunConv(conv,RELU_EN,0,image,W_conv1,b_conv1,h_conv1)
#                     print(h_conv1)
                    t7 = time.time()
        #             print(t7-t2)
                    RunConv(conv,RELU_EN,0,h_conv1,W_conv2,b_conv2,h_conv2)

                    RunConv(conv,RELU_EN,0,h_conv2,W_conv3,b_conv3,h_conv3)

                    RunConv(conv,RELU_EN,0,h_conv3,W_conv4,b_conv4,h_conv4)

                    RunConv(conv,RELU_EN,0,h_conv4,W_conv5,b_conv5,h_conv5)

                    RunConv(conv,RELU_EN,0,h_conv5,W_conv6,b_conv6,h_conv6)

                    RunConv(conv,RELU_EN,0,h_conv6,W_conv7,b_conv7,h_conv7)

                    RunConv(conv,RELU_EN,0,h_conv7,W_conv8,b_conv8,h_conv8)

                    RunConv(conv,RELU_EN,0,h_conv8,W_conv9,b_conv9,h_conv9)

                    RunConv(conv,0,1,h_conv9,W_conv10,b_conv10,h_conv10)
                    
#                     img_final[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)] = sigmid_all(h_conv10[:,:,0],32,32)
                    decision[:,:] = h_conv10[:,:,0]
                    Fusion(fusion,image1,image2,decision,fus)
                    t3 = time.time()
                    img_final[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1),:] = fus
        #         t4 = time.time()
            t5  = time.time()
        # print(t1-t0,'...',t2-t1,'....',t3-t2,'...',t4-t3)
#             print(img_final)
            print(t5-t6)

            #g=input("input enter to continue")
            cv2.imwrite('./picture_out/'+name_out,img_final)