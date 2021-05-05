# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:22:52 2021

@author: JerryDai
"""
# In[] import package

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABCMeta, abstractmethod
import cv2
import pandas as pd
import time
from tqdm import tqdm

# In[] def

def load_pic_data_by_txt(pic_load_list): # pic_load_list = list(train_txt.loc[batch_index, 'pic_path'])
    # tmp_pic_w, tmp_pic_h, tmp_pic_d =  cv2.imread(pic_load_list[0]).shape
    tmp_pic_w, tmp_pic_h=  cv2.imread(pic_load_list[0], cv2.IMREAD_GRAYSCALE).shape
    # tmp_size = min(tmp_pic_w, tmp_pic_h)
    tmp_size = 64
    batch_size = len(pic_load_list)
    # tmp_x_array = np.zeros(batch_size * tmp_pic_d * tmp_size * tmp_size).reshape((batch_size, tmp_pic_d, tmp_size, tmp_size))
    # tmp_x_array = np.zeros(batch_size * 1 * tmp_size * tmp_size).reshape((batch_size, 1, tmp_size, tmp_size))
    tmp_x_array = np.zeros(batch_size * 1 * 32 * 32).reshape((batch_size, 1, 32, 32))
    # tmp_x_array = np.array([])
        
    for index, tmp_pic_path in enumerate(pic_load_list):
        # print(index)
        # print(tmp_pic_path)
        
        # temp_pic = cv2.imread(pic_load_list[index])
        temp_pic = cv2.imread(pic_load_list[index], cv2.IMREAD_GRAYSCALE)
        temp_pic = crop_pic(temp_pic, tmp_size)
        temp_pic = cv2.resize(temp_pic, (32, 32), interpolation=cv2.INTER_CUBIC)
        # temp_pic = pic_channel_reshape(temp_pic)
        # temp_pic = np.mean(temp_pic, axis = 0).reshape((1, tmp_size, tmp_size))
        tmp_x_array[index] = temp_pic
        # tmp_x_array = np.append(tmp_x_array, temp_pic, axis = 1)
    
    tmp_x_array.astype('float32')
    tmp_x_array /= 255
    
    return tmp_x_array

def MakeOneHot(Y, D_out): # Y = Y_batch
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def crop_pic(image, size): # image = temp_pic size = 64
    # h, w, d = image.shape
    h, w = image.shape
    x, y = w // 2, h // 2
    block_size = size // 2
    # crop_image = image[y - block_size : y + block_size, x - block_size : x + block_size, :]
    crop_image = image[y - block_size : y + block_size, x - block_size : x + block_size]
    
    return crop_image

def pic_channel_reshape(image): # image = crop_img
    h, w, d = image.shape
    rechannel_image = np.append(np.append(image[:,:,0], image[:,:,1]), image[:,:,2]).reshape(d, h, w)
    return rechannel_image

def get_batch_data(train_txt, batch_size, number_of_category):
    number_of_data = len(train_txt)
    # batch_index = random.randint(0, number_of_data - batch_size)
    batch_index = np.random.randint(number_of_data, size = batch_size)
    # batch_x = load_pic_data_by_txt(list(train_txt.loc[batch_index : batch_index + batch_size - 1, 'pic_path']))
    batch_x = load_pic_data_by_txt(list(train_txt.loc[batch_index, 'pic_path']))
    # batch_y = MakeOneHot(np.array(train_txt.loc[batch_index : batch_index + batch_size - 1, 'label']), number_of_category)
    batch_y = MakeOneHot(np.array(train_txt.loc[batch_index, 'label']), number_of_category)
    
    return batch_x, batch_y

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

# def get_accuracy(test_txt, batch_size):
#     total = len(y_test)
#     for i in range(len(test_txt) / batch_size)
#     tmp_txt = test_txt
    
#     x_test, y_test = get_batch_data(test_txt, )
#     count = 0
    
    
#     accuracy = count / total
    
#     return accuracy

class FC():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        #print("Build FC")
        self.cache = None
        #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        #print("FC: _forward")
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        #print("FC: _backward")
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        #self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        # Update the parameters
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']

class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def _forward(self, X): # X = h4
        #print("ReLU: _forward")
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX

class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        self.cache = X
        return 1 / (1 + np.exp(-X))

    def _backward(self, dout):
        X = self.cache
        dX = dout*X*(1-X)
        return dX

class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        #print("Softmax: _forward")
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX

class Dropout():
    """
    Dropout layer
    """
    def __init__(self, p=1):
        self.cache = None
        self.p = p

    def _forward(self, X):
        M = (np.random.rand(*X.shape) < self.p) / self.p
        self.cache = X, M
        return X*M

    def _backward(self, dout):
        X, M = self.cache
        dX = dout*M/self.p
        return dX

class Conv():
    """
    Conv layer
    """
    def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
        self.Cin = Cin
        self.Cout = Cout
        self.F = F
        self.S = stride
        #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.cache = None
        self.pad = padding

    def _forward(self, X): # X = p2 Cin = 16 Cout = 120 F = 3 pad = 1 S = 1
        # X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant') # X = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
        
        tmp_X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
        
        (N, Cin, H, W) = X.shape # (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1 # H_ = H - F + 1  # H_ = int((H - F + 2 * pad) / S + 1)
        W_ = W - self.F + 1 # W_ = W - F + 1  # W_ = int((W - F + 2 * pad) / S + 1)
        Y = np.zeros((N, self.Cout, H_, W_)) # Y = np.zeros((N, Cout, H_, W_))

        for n in range(N):
            for c in range(self.Cout):
                for h in range(H_):
                    for w in range(W_):
                        Y[n, c, h, w] = np.sum(tmp_X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]
        
        # W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0}
        # b = {'val': np.random.randn(Cout), 'grad': 0}
        # for n in range(N):
        #     for c in range(Cout):
        #         for h in range(H_):
        #             for w in range(W_):
        #                 Y[n, c, h, w] = np.sum(X[n, :, h:h+F, w:w+F] * W['val'][c, :, :, :]) + b['val'][c]
        
        
        self.cache = tmp_X
        return Y

    def _backward(self, dout):
        # dout (N,Cout,H_,W_)
        # W (Cout, Cin, F, F)
        X = self.cache
        (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1 # H_ = int((H - F + 2 * pad) / S + 1) - 2 * pad
        W_ = W - self.F + 1 # W_ = int((W - F + 2 * pad) / S + 1) - 2 * pad
        W_rot = np.rot90(np.rot90(self.W['val'])) # W_rot = np.rot90(np.rot90(W['val']))

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['val'].shape) # dW = np.zeros(W['val'].shape)
        db = np.zeros(self.b['val'].shape)

        # dW
        for co in range(self.Cout):
            for ci in range(Cin):
                for h in range(self.F):
                    for w in range(self.F):
                        dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])
        # dW Test
        # for co in range(Cout):
        #     for ci in range(Cin):
        #         for h in range(F):
        #             for w in range(F):
        #                 dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_].shape * dout[:,co,:,:].shape)   
        
        # db
        for co in range(self.Cout):
            db[co] = np.sum(dout[:,co,:,:])

        dout_pad = np.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
        #print("dout_pad.shape: " + str(dout_pad.shape))
        # dX
        for n in range(N):
            for ci in range(Cin):
                for h in range(H):
                    for w in range(W):
                        #print("self.F.shape: %s", self.F)
                        #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                        dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

        return dX

class MaxPool():
    def __init__(self, F, stride):
        self.F = F
        self.S = stride
        self.cache = None

    def _forward(self, X): # X = a3
        # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
        (N,Cin,H,W) = X.shape
        F = self.F # F = 2
        W_ = int(float(W)/F)
        H_ = int(float(H)/F)
        Y = np.zeros((N,Cin,W_,H_))
        M = np.zeros(X.shape) # mask
        for n in range(N):
            for cin in range(Cin):
                for w_ in range(W_):
                    for h_ in range(H_):
                        Y[n,cin,w_,h_] = np.max(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)])
                        i,j = np.unravel_index(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(), (F,F))
                        M[n,cin,F*w_+i,F*h_+j] = 1
        self.cache = M
        return Y

    def _backward(self, dout):
        M = self.cache
        (N,Cin,H,W) = M.shape
        dout = np.array(dout)
        #print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
        dX = np.zeros(M.shape)
        for n in range(N):
            for c in range(Cin):
                #print("(n,c): (%s,%s)" % (n,c))
                dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
        return dX*M

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        # softmax = Softmax()
        # prob = softmax._forward(Y_pred) # Y_pred = y_pred A=softmax._forward(Y_pred)
        prob = Y_pred
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        loss = NLLLoss(Y_pred, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = Y_pred.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv(1, 6, 5, padding=0)
        self.ReLU1 = Sigmoid()
        self.pool1 = MaxPool(2, 2)
        self.conv2 = Conv(6, 16, 5, padding=0)
        self.ReLU2 = Sigmoid()
        self.pool2 = MaxPool(2, 2)
        self.FC1 = FC(400, 128)
        # self.conv3 = Conv(16, 120, 5)
        self.ReLU3 = Sigmoid()
        self.FC2 = FC(128, 100)
        # self.FC1 = FC(120*57*57, 128)
        self.ReLU4 = Sigmoid()
        self.FC3 = FC(100, 50)
        # self.FC2 = FC(128, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        
        h2 = self.conv2._forward(p1)
        a2 = self.ReLU2._forward(h2)
        p2 = self.pool2._forward(a2)
        
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten
        h3 = self.FC1._forward(fl)
        a3 = self.ReLU3._forward(h3)
        
        h4 = self.FC2._forward(a3)
        a4 = self.ReLU4._forward(h4)
        
        h5 = self.FC3._forward(a4)
        a5 = self.Softmax._forward(h5)
        return a5

    def backward(self, dout):
        #dout = self.Softmax._backward(dout)
        dout = self.FC3._backward(dout)
        # dout = self.FC2._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.FC2._backward(dout)
        # dout = self.FC1._backward(dout)
        dout = self.ReLU3._backward(dout)
        dout = self.FC1._backward(dout)
        # dout = self.conv3._backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]
        # return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params

class Improved_LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv(1, 6, 5)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2, 2)
        
        self.conv2 = Conv(6, 16, 5)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2, 2)
        
        # self.FC1 = FC(3136 , 128)
        # self.conv3 = Conv(16, 120, 5)
        # self.ReLU3 = ReLU()
        # self.pool3 = MaxPool(2, 2)
        
        self.FC1 = FC(400, 128)
        self.ReLU4 = ReLU()
        
        self.FC2 = FC(128, 100)
        self.ReLU5 = ReLU()
        
        self.FC3 = FC(100, 50)
        self.Softmax = Softmax()

        self.p3_shape = None

    def forward(self, X): # X = batch_x
        h1 = self.conv1._forward(X) # h1 = conv1._forward(X)
        a1 = self.ReLU1._forward(h1) # a1 = ReLU1._forward(h1)
        p1 = self.pool1._forward(a1) # p1 = pool1._forward(a1)
        
        h2 = self.conv2._forward(p1) # h2 = conv2._forward(p1)
        a2 = self.ReLU2._forward(h2) # a2 = ReLU2._forward(h2)
        p2 = self.pool2._forward(a2) # p2 = pool2._forward(a2)
        
        # h3 = self.conv3._forward(p2) # h3 = conv3._forward(p2)
        # a3 = self.ReLU3._forward(h3) # a3 = ReLU3._forward(h3)
        # p3 = self.pool3._forward(a3) # p3 = pool3._forward(a3)
        
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten
        h4 = self.FC1._forward(fl) # h4 = FC1._forward(fl)
        # h3 = self.conv3._forward(fl)
        a4 = self.ReLU4._forward(h4) # a4 = ReLU4._forward(h4)
        
        h5 = self.FC2._forward(a4) # h5 = FC2._forward(a4)
        # h4 = self.FC1._forward(a3)        
        a5 = self.ReLU5._forward(h5) # a5 = ReLU5._forward(h5)
        
        h6 = self.FC3._forward(a5) # h6 = FC3._forward(a5)
        # h5 = self.FC2._forward(a5)
        a6 = self.Softmax._forward(h6) # a6 = Softmax._forward(h6)
        return a6

    def backward(self, dout):
        # dout = self.Softmax._backward(dout)
        
        dout = self.FC3._backward(dout) # dout = FC3._backward(dout)
        # dout = self.FC2._backward(dout)
        dout = self.ReLU5._backward(dout) # dout = ReLU5._backward(dout)
        dout = self.FC2._backward(dout) # dout = FC2._backward(dout)
        # dout = self.FC1._backward(dout)
        dout = self.ReLU4._backward(dout) # dout = ReLU4._backward(dout)
        dout = self.FC1._backward(dout) # dout = FC1._backward(dout)
        # dout = self.conv3._backward(dout)
        dout = dout.reshape(self.p2_shape) # dout = dout.reshape(p3_shape)
        
        # dout = self.pool3._backward(dout) # dout = pool3._backward(dout)
        # dout = self.ReLU3._backward(dout) # dout = ReLU3._backward(dout)
        # dout = self.conv3._backward(dout) # dout = conv3._backward(dout)
        
        dout = self.pool2._backward(dout) # dout = pool2._backward(dout)
        dout = self.ReLU2._backward(dout) # dout = ReLU2._backward(dout)
        dout = self.conv2._backward(dout) # dout = conv2._backward(dout)
        
        dout = self.pool1._backward(dout) # dout = pool1._backward(dout)
        dout = self.ReLU1._backward(dout) # dout = ReLU1._backward(dout)
        dout = self.conv1._backward(dout) # dout = conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]
        # return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params
        # [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def step(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])
            
class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])


# In[] Load Data

batch_size = 8
number_of_category = 50

train_txt = pd.read_csv('train.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
val_txt = pd.read_csv('val.txt', header = None, names = ['pic_path', 'label'], sep = ' ')
test_txt = pd.read_csv('test.txt', header = None, names = ['pic_path', 'label'], sep = ' ')

# train_x, train_y = get_batch_data(train_txt, len(train_txt), number_of_category)
val_x, val_y = get_batch_data(val_txt, len(val_txt), number_of_category)
test_x, test_y = get_batch_data(test_txt, len(test_txt), number_of_category)


# In[] LeNet5
# model = LeNet5()

# losses = []
# # optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
# optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
# # optim = SGD(model.get_params(), lr=0.001)
# criterion = CrossEntropyLoss()

# # TRAIN
# ITER = 2000
# for i in range(ITER):
#     print(i)
#     # get batch, make onehot
#     start = time.time()
#     batch_x, batch_y = get_batch_data(train_txt, batch_size, number_of_category)
#     end = time.time()
#     print("get batch data time：%f sec" % (end - start))

#     # forward, loss, backward, step
#     start = time.time()
#     pred_y = model.forward(batch_x)
#     loss, dout = criterion.get(pred_y, batch_y)
#     model.backward(dout)
#     optim.step()
#     end = time.time()
#     print("update weights time：%f sec" % (end - start))
    
#     if np.isnan(loss):
#         break
#     if i % 1 == 0:
#         print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
#         losses.append(loss)

# draw_losses(losses)
# pd.DataFrame(losses).to_csv('LeNet5_losses.csv')

# # save weights
# weights = model.get_params()
# pd.DataFrame(weights).to_csv('LeNet5_model_weights.csv')

# # load weights
# # tmp_weight_df = pd.read_csv('LeNet5_model_weights.csv')
# # weights = list()
# # for i in range(len(tmp_weight_df)):
# #     weights.append(dict(tmp_weight_df.iloc[i, 1:3]))

# # # set weights
# # model.set_params(weights)


# # TRAIN SET ACC
# # Y_pred = model.forward(X_train)
# # result = np.argmax(Y_pred, axis=1) - Y_train
# # result = list(result)
# # print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# result_list = list()
# # VAL SET ACC
# eval_batch_size = 8
# eval_iter = int(np.ceil(len(val_x) / eval_batch_size))
# count = 0
# for i in tqdm(range(eval_iter)):
#     print(i)
#     if (i + 1) * batch_size > len(val_x):
#         tmp_val_x = val_x[i : -1]
#         tmp_val_y = val_y[i : -1]
#     else:
#         tmp_val_x = val_x[i : (i + 1) * batch_size]
#         tmp_val_y = val_y[i : (i + 1) * batch_size]
    
#     y_pred = model.forward(tmp_val_x)
#     result = np.argmax(y_pred, axis=1) - np.argmax(tmp_val_y, axis=1)
#     count += list(result).count(0)
    
# print("Val--> Correct: " + str(count) + " out of " + str(val_x.shape[0]) + ", acc=" + str(count/val_x.shape[0]))
# result_list.append(count)

# # TEST SET ACC
# eval_iter = int(np.floor(len(test_x) / batch_size))
# count = 0
# for i in tqdm(range(eval_iter)):
#     print(i)
#     if (i + 1) * batch_size > len(val_x):
#         tmp_test_x = test_x[i : -1]
#         tmp_test_y = test_y[i : -1]
#     else:
#         tmp_test_x = test_x[i : (i + 1) * batch_size]
#         tmp_test_y = test_y[i : (i + 1) * batch_size]
    
#     y_pred = model.forward(tmp_test_x)
#     result = np.argmax(y_pred, axis=1) - np.argmax(tmp_test_y, axis=1)
#     count += list(result).count(0)
# print("TEST--> Correct: " + str(count) + " out of " + str(test_x.shape[0]) + ", acc=" + str(count/test_x.shape[0]))
# result_list.append(count)

# pd.DataFrame(result_list).to_csv('LeNet5_model_result.csv')

# In[] Improved_LeNet5

model = Improved_LeNet5()

losses = []
#optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
# optim = SGD(model.get_params(), lr=0.001)
criterion = CrossEntropyLoss()

# TRAIN
ITER = 7915
for i in range(ITER):
    print(i)
    # get batch, make onehot
    start = time.time()
    batch_x, batch_y = get_batch_data(train_txt, batch_size, number_of_category)
    end = time.time()
    print("get batch data time：%f sec" % (end - start))

    # forward, loss, backward, step
    start = time.time()
    pred_y = model.forward(batch_x)
    loss, dout = criterion.get(pred_y, batch_y)
    model.backward(dout)
    optim.step()
    end = time.time()
    print("update weights time：%f sec" % (end - start))

    if np.isnan(loss):
        break
    if i % 791 == 0:
        print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
        losses.append(loss)

draw_losses(losses)
pd.DataFrame(losses).to_csv('Improved_LeNet5_losses.csv')

# save weights
weights = model.get_params()
pd.DataFrame(weights).to_csv('Improved_LeNet5_model_weights.csv')

# load weights
# tmp_weight_df = pd.read_csv('Improved_LeNet5_model_weights.csv')
# weights = list()
# for i in range(len(tmp_weight_df)):
#     weights.append(dict(tmp_weight_df.iloc[i, 1:3]))

# # set weights
# model.set_params(weights)


# TRAIN SET ACC
# Y_pred = model.forward(X_train)
# result = np.argmax(Y_pred, axis=1) - Y_train
# result = list(result)
# print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

result_list = list()
# VAL SET ACC
eval_batch_size = 8
eval_iter = int(np.ceil(len(val_x) / eval_batch_size))
count = 0
for i in tqdm(range(eval_iter)):
    print(i)
    if (i + 1) * batch_size > len(val_x):
        tmp_val_x = val_x[i : -1]
        tmp_val_y = val_y[i : -1]
    else:
        tmp_val_x = val_x[i : (i + 1) * batch_size]
        tmp_val_y = val_y[i : (i + 1) * batch_size]
    
    y_pred = model.forward(tmp_val_x)
    result = np.argmax(y_pred, axis=1) - np.argmax(tmp_val_y, axis=1)
    count += list(result).count(0)
print("Val--> Correct: " + str(count) + " out of " + str(val_x.shape[0]) + ", acc=" + str(count/val_x.shape[0]))
result_list.append(count)

# TEST SET ACC
eval_iter = int(np.floor(len(test_x) / batch_size))
count = 0
for i in tqdm(range(eval_iter)):
    print(i)
    if (i + 1) * batch_size > len(val_x):
        tmp_test_x = test_x[i : -1]
        tmp_test_y = test_y[i : -1]
    else:
        tmp_test_x = test_x[i : (i + 1) * batch_size]
        tmp_test_y = test_y[i : (i + 1) * batch_size]
    
    y_pred = model.forward(tmp_test_x)
    result = np.argmax(y_pred, axis=1) - np.argmax(tmp_test_y, axis=1)
    count += list(result).count(0)
print("TEST--> Correct: " + str(count) + " out of " + str(test_x.shape[0]) + ", acc=" + str(count/test_x.shape[0]))
result_list.append(count)

pd.DataFrame(result_list).to_csv('Improved_LeNet5_model_result.csv')

# In[] Plot Triaining Curve
losses_lenet5 = pd.read_csv('LeNet5_losses.csv', index_col=0)
improve_losses_lenet5 = pd.read_csv('Improved_LeNet5_losses.csv', index_col=0)

plt.plot(losses_lenet5, label = 'Lenet5')
plt.plot(improve_losses_lenet5, label = 'Improved_LeNet5')
plt.title('Triaining Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png')
plt.close()

# In[] Plot Accuracy
result_lenet5 = pd.read_csv('LeNet5_model_result.csv', index_col=0)
result_losses_lenet5 = pd.read_csv('Improved_LeNet5_model_result.csv', index_col=0)

result = pd.concat([result_lenet5, result_losses_lenet5], axis = 0) / 450

result.columns = ['Accuracy']
model_list = list(['LeNet5', 'LeNet5', 'Improved_LeNet5', 'Improved_LeNet5'])
result['Model'] = model_list
dataname_list = list(['Val', 'Test', 'Val', 'Test'])
result['Data'] = dataname_list

result = result.reset_index(drop=True)
sns.barplot(x = "Model", y = 'Accuracy', hue = 'Data', data = result)

plt.savefig('Accuracy.png')
plt.close()

# In[] init

# conv1 = Conv(1, 6, 5)
# ReLU1 = ReLU()
# pool1 = MaxPool(2, 2)

# conv2 = Conv(6, 16, 5)
# ReLU2 = ReLU()
# pool2 = MaxPool(2, 2)

# # .FC1 = FC(3136 , 128)
# conv3 = Conv(16, 120, 5)
# ReLU3 = ReLU()
# pool3 = MaxPool(2, 2)

# FC1 = FC(1920 , 128)
# ReLU4 = ReLU()

# FC2 = FC(128, 100)
# ReLU5 = ReLU()

# FC3 = FC(100, 50)
# Softmax = Softmax()

# p3_shape = None

# In[] Forward
# batch_x, batch_y = get_batch_data(train_txt, batch_size, number_of_category)

# X = batch_x
# h1 = conv1._forward(X)
# a1 = ReLU1._forward(h1)
# p1 = pool1._forward(a1)

# h2 = conv2._forward(p1)
# a2 = ReLU2._forward(h2)
# p2 = pool2._forward(a2)

# h3 = conv3._forward(p2)
# a3 = ReLU3._forward(h3)
# p3 = pool3._forward(a3)

# p3_shape = p3.shape
# fl = p3.reshape(X.shape[0],-1) # Flatten
# h4 = FC1._forward(fl)
# a4 = ReLU4._forward(h4)

# h5 = FC2._forward(a4)
# a5 = ReLU5._forward(h5)

# h6 = FC3._forward(a5)
# a6 = Softmax._forward(h6)

# In[] Backward
# pred_y = model.forward(batch_x)
# loss, dout = criterion.get(pred_y, batch_y)

# dout = FC3._backward(dout)
# dout = ReLU5._backward(dout)
# dout = FC2._backward(dout)
# dout = ReLU4._backward(dout)
# dout = FC1._backward(dout)
# dout = dout.reshape(p3_shape)

# dout = pool3._backward(dout)
# dout = ReLU3._backward(dout) # dout_backup = dout
# dout = conv3._backward(dout) # dout = dout_backup

# dout = pool2._backward(dout)
# dout = ReLU2._backward(dout)
# dout = conv2._backward(dout)

# dout = pool1._backward(dout)
# dout = ReLU1._backward(dout)
# dout = conv1._backward(dout)


