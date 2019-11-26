from pynq import Xlnk
from pynq import Overlay
import numpy as np
import pynq.lib.dma
import cv2
import time
overlay = Overlay('final_test4.bit')   
overlay.download()
dma = overlay.axi_dma_0          
xlnk = Xlnk()
print('overlay over')

def sigmoid_all(x,h,w):
    s = np.zeros((h,w))
    for hh in range(h):
        for ww in range(w):
            s[hh][ww] = 1 / (1 + np.exp(-x[hh][ww]))
    return s

IN_HEIGHT = 32
IN_WIDTH = 32
IN_CH = 2
OUT_CH = 32
OUT_HEIGHT = 32
OUT_WIDTH =32
KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
OUT = 1


#Initialize W, bias
W_conv1 = np.zeros((KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH))
W_conv10 = np.zeros((KERNEL_HEIGHT,KERNEL_WIDTH,OUT_CH,OUT_CH))
b_conv10 = np.zeros((OUT_CH))
w_conv1=np.load('./weights_npy/conv1_weights.npy')
W_conv1[:,:,0:2,:]=w_conv1[:,:,:,:]
B_conv1=np.load('./weights_npy/conv1_bias.npy')

w_conv2=np.load('./weights_npy/conv2_weights.npy')
B_conv2=np.load('./weights_npy/conv2_bias.npy')

w_conv2=np.load('./weights_npy/conv2_weights.npy')

B_conv2=np.load('./weights_npy/conv2_bias.npy')


w_conv3=np.load('./weights_npy/conv3_weights.npy')

B_conv3=np.load('./weights_npy/conv3_bias.npy')


w_conv4=np.load('./weights_npy/conv4_weights.npy')

B_conv4=np.load('./weights_npy/conv4_bias.npy')


w_conv5=np.load('./weights_npy/conv5_weights.npy')

B_conv5=np.load('./weights_npy/conv5_bias.npy')


w_conv6=np.load('./weights_npy/conv6_weights.npy')

B_conv6=np.load('./weights_npy/conv6_bias.npy')

w_conv7=np.load('./weights_npy/conv7_weights.npy')

B_conv7=np.load('./weights_npy/conv7_bias.npy')


w_conv8=np.load('./weights_npy/conv8_weights.npy')

B_conv8=np.load('./weights_npy/conv8_bias.npy')

w_conv9=np.load('./weights_npy/conv9_weights.npy')

B_conv9=np.load('./weights_npy/conv9_bias.npy')


w_conv10=np.load('./weights_npy/conv10_weights.npy')
W_conv10[:,:,:,0:1]=w_conv10[:,:,:,:]
B_conv10=np.load('./weights_npy/conv10_bias.npy')
b_conv10[0:1]=B_conv10[0:1]
precision = 1000
W1 = np.multiply(W_conv1.flatten(),precision)
B1 = np.multiply(B_conv1.flatten(),precision)

W2 = np.multiply(w_conv2.flatten(),precision)
B2 = np.multiply(B_conv2.flatten(),precision)

W3 = np.multiply(w_conv3.flatten(),precision)
B3 = np.multiply(B_conv3.flatten(),precision)

W4 = np.multiply(w_conv4.flatten(),precision)
B4 = np.multiply(B_conv4.flatten(),precision)

W5 = np.multiply(w_conv5.flatten(),precision)
B5 = np.multiply(B_conv5.flatten(),precision)

W6 = np.multiply(w_conv6.flatten(),precision)
B6 = np.multiply(B_conv6.flatten(),precision)

W7 = np.multiply(w_conv7.flatten(),precision)
B7 = np.multiply(B_conv7.flatten(),precision)

W8 = np.multiply(w_conv8.flatten(),precision)
B8 = np.multiply(B_conv8.flatten(),precision)

W9 = np.multiply(w_conv9.flatten(),precision)
B9 = np.multiply(B_conv9.flatten(),precision)

W10 = np.multiply(W_conv10.flatten(),precision)
B10 = np.multiply(b_conv10.flatten(),precision)

print("Finish initial")
t1=time.time()

in_buffer = xlnk.cma_array(shape=(3000,), dtype=np.int32)
Configuration = xlnk.cma_array(shape=(3,), dtype=np.int32)
in_buffer_x = xlnk.cma_array(shape=(((32*32*32) % 3000),), dtype=np.int32)
in_buffer_bias = xlnk.cma_array(shape=(((B1.shape[0]) % 3000),), dtype=np.int32)
in_buffer_weight = xlnk.cma_array(shape=(((W1.shape[0]) % 3000),), dtype=np.int32)
in_buffer_param = xlnk.cma_array(shape=(3,), dtype=np.int32)
in_buffer_out = xlnk.cma_array(shape=(((32*32*32) % 3000),), dtype=np.int32)
t2 = time.time()

def conv(dma,xlnk,in_buffer,relu,sigmoid,precision,x,Bias,Weight,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out):
    t1 = time.time()
    Configuration[0] = relu
    Configuration[1] = sigmoid
    Configuration[2] = precision
    dma.sendchannel.transfer(Configuration)
    dma.sendchannel.wait()
    dma.recvchannel.transfer(Configuration)
    dma.recvchannel.wait()
#     print(Configuration)
    for idx in range(int(x.shape[0] / 3000)):
        temproray = x[(idx * 3000):(idx * 3000) + 3000]
        np.copyto(in_buffer, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer)
        dma.sendchannel.wait()

    if ((x.shape[0]) % 3000 != 0):
        temproray = x[int((x.shape[0]) / 3000) * 3000:x.shape[0]]
        np.copyto(in_buffer_x, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer_x)
        dma.sendchannel.wait()

    for idx in range(int(Bias.shape[0] / 3000)):
        temproray = Bias[(idx * 3000):(idx * 3000) + 3000]
        np.copyto(in_buffer, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer)
        dma.sendchannel.wait()

        
    if ((Bias.shape[0]) % 3000 != 0):
        temproray = Bias[int((Bias.shape[0]) / 3000) * 3000:Bias.shape[0]]
        np.copyto(in_buffer_bias, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer_bias)
        dma.sendchannel.wait()

        
    for idx in range(int(Weight.shape[0] / 3000)):
        temproray = Weight[(idx * 3000):(idx * 3000) + 3000]
        np.copyto(in_buffer, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer)
        dma.sendchannel.wait()
        
        
    if ((Weight.shape[0]) % 3000 != 0):
        temproray = Weight[int((Weight.shape[0]) / 3000) * 3000:Weight.shape[0]]
        np.copyto(in_buffer_weight, temproray, casting='unsafe')
        dma.sendchannel.transfer(in_buffer_weight)
        dma.sendchannel.wait()
        
    dma.sendchannel.transfer(in_buffer_param)
    dma.sendchannel.wait()
    dma.recvchannel.transfer(in_buffer_param)
    dma.recvchannel.wait()
    Total_result = in_buffer_param[0]
    Out_H = in_buffer_param[1]
    Out_W = in_buffer_param[2]
#     print(Total_result,Out_H,Out_W)
    Output = np.zeros((Total_result,))
    # Return Final size from FPGA
    
    for idx in range(int((Total_result) / 3000)):
        dma.sendchannel.transfer(in_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(in_buffer)
        dma.recvchannel.wait()
        np.copyto(Output[(idx * 3000):((idx * 3000) + 3000)], in_buffer, casting='unsafe')
#         print('out:',in_buffer)
    if ((Total_result) % 3000 != 0):
        in_buffer_out = xlnk.cma_array(shape=(((Total_result) % 3000),), dtype=np.int32)
        dma.sendchannel.transfer(in_buffer_out)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(in_buffer_out)
        dma.recvchannel.wait()
        np.copyto(Output[(int((Total_result) / 3000) * 3000):Total_result], in_buffer_out, casting='unsafe')
#         print('x:',in_buffer_out)
    t2 = time.time()
    print(t2-t1)
#     channel = int(Total_result/(Out_H*Out_W))
#     print(channel)
    # Return result to float format and reshape
#     Output = np.divide(Output, precision).reshape(Out_H, Out_W,channel)
#     xlnk.xlnk_reset()
    return Output

print('def over')
print(t2-t1)

img1=cv2.imread("./testdata/01-A.jpg",cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2=cv2.imread("./testdata/01-B.jpg",cv2.IMREAD_GRAYSCALE).astype(np.float32)
print("Read image")
#image1=image1.reshape((IN_HEIGHT1,IN_WIDTH1,IN_CH1))
inputsize = 32
imgsize = 256    
img1=cv2.resize(img1,(imgsize,imgsize)) 
img2=cv2.resize(img2,(imgsize,imgsize))
image=np.zeros((inputsize,inputsize,inputsize))
img_final = np.zeros((imgsize,imgsize))
t1= time.time()
for h in range(int(imgsize/inputsize)):
    for w in range(int(imgsize/inputsize)):
#         t0 =time.time()
        img1_crop = img1[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)]
        img2_crop = img2[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)]
        
        
#         imageyy = np.zeros((32,32,32))
#         print(inputsize*h,inputsize*(h+1))
#         print(inputsize*w,inputsize*(w+1))
#         t1 =time.time()
        image[:,:,0] = img1_crop/255.0
        image[:,:,1] = img2_crop/255.0
        img = np.multiply(image.flatten(),precision)
        out1 = conv(dma,xlnk,in_buffer,1,0,precision,img,B1,W1,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        # test =np.divide(out1,precision).reshape(32,32,32)
        # print(test[:,:,0])
        out2 = conv(dma,xlnk,in_buffer,1,0,precision,out1,B2,W2,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        # test =np.divide(out2,precision).reshape(32,32,32)
        # print(test[:,:,0])
        out3 = conv(dma,xlnk,in_buffer,1,0,precision,out2,B3,W3,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        # test =np.divide(out3,precision).reshape(32,32,32)
        # print(test[:,:,0])
        out4 = conv(dma,xlnk,in_buffer,1,0,precision,out3,B4,W4,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        # test =np.divide(out4,precision).reshape(32,32,32)
        # print(test[:,:,0])
        out5 = conv(dma,xlnk,in_buffer,1,0,precision,out4,B5,W5,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        out6 = conv(dma,xlnk,in_buffer,1,0,precision,out5,B6,W6,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        out7 = conv(dma,xlnk,in_buffer,1,0,precision,out6,B7,W7,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        out8 = conv(dma,xlnk,in_buffer,1,0,precision,out7,B8,W8,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        out9 = conv(dma,xlnk,in_buffer,1,0,precision,out8,B9,W9,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        # test =np.divide(out9,precision).reshape(32,32,32)
        # print(test[:,:,0])
        out = conv(dma,xlnk,in_buffer,0,1,precision,out9,B10,W10,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out)
        final = np.divide(out,precision).reshape(32,32,32)
        img_final[inputsize*h:inputsize*(h+1),inputsize*w:inputsize*(w+1)] = final[:,:,0]
t3 = time.time()
print(t3-t1)
x = sigmoid_all(img_final,256,256)  
final = img1*x+img2*(1-x)
# conv(dma,xlnk,in_buffer,relu,sigmoid,x,Bias,Weight,Configuration,in_buffer_x,in_buffer_bias,in_buffer_weight,in_buffer_param,in_buffer_out):
t2 = time.time()
print(t2-t1)
cv2.imwrite('./dma.jpg',x*255)