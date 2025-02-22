# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:28:55 2021

@author: JerryDai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[] 

LeNet_Pytorch_loss = pd.read_csv('Pytorch_LeNet5_losses.csv',index_col=0)
Tensor_LeNet_loss = pd.read_csv('Tensor_LeNet_loss.csv',index_col=0)
Imporved_LeNet_loss = pd.read_csv('Improved_LeNet5_losses.csv',index_col=0)

LeNet_Pytorch_result = pd.read_csv('Pytorch_LeNet5_model_result.csv',index_col=0)
Tensor_LeNet_result = pd.read_csv('Tensor_LeNet_model_result.csv',index_col=0)
Imporved_LeNet_result = pd.read_csv('Improved_LeNet5_model_result.csv',index_col=0)

LeNet_Pytorch_accuracy = LeNet_Pytorch_result / 450
Tensor_LeNet_accuracy = Tensor_LeNet_result
Imporved_LeNet_accuracy = Imporved_LeNet_result / 450

# In[] loss

plt.plot(LeNet_Pytorch_loss, label = 'Pytorch LeNet5')
plt.plot(Tensor_LeNet_loss, label = 'Tensorflow LeNet5')
plt.plot(Imporved_LeNet_loss[0 : -1], label = 'Improved LeNet5')
plt.legend()

plt.title('Training Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.savefig('Loss.png')
plt.close()
# In[] accuracy
totol_accuracy = pd.concat([LeNet_Pytorch_accuracy, Tensor_LeNet_accuracy], axis = 0)
totol_accuracy = pd.concat([totol_accuracy, Imporved_LeNet_accuracy], axis = 0)

model_list = list(['Pytorch_LeNet5', 'Pytorch_LeNet5', 'Tensor_LeNet5', 'Tensor_LeNet5', 'Improved_LeNet5', 'Improved_LeNet5'])
totol_accuracy['Model'] = model_list

dataname_list = list(['Val', 'Test', 'Val', 'Test', 'Val', 'Test'])
totol_accuracy['Data'] = dataname_list

totol_accuracy = totol_accuracy.reset_index(drop=True)
totol_accuracy = totol_accuracy.rename(columns={'0' : 'Accuracy'})

sns.barplot(x = "Model", y = 'Accuracy', hue = 'Data', data = totol_accuracy)

plt.savefig('Accuracy.png')
plt.close()
