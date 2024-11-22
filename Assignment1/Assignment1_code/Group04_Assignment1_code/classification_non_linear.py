# -*- coding: utf-8 -*-
"""DL_ASSIG_1.2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sj0J4dJ4qgO5MGXr9FDDt_tp8wkHZoIS
"""

from google.colab import drive
drive.mount('/content/drive')

"""**Importing the libraries**"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg

"""**Getting the data file path**"""

data_path = "/content/drive/MyDrive/Group04/Classification/NLS_Group04.txt"

"""**Reading the input data**"""

data = pd.read_csv(data_path, skiprows=1, header=None, delimiter=" ")
data = pd.DataFrame(data)
data = data.drop([2], axis=1)

class1 = data.iloc[:300]
class2 = data.iloc[301:500]
class3 = data.iloc[501:1000]

plt.scatter(class1[0], class1[1])
plt.scatter(class2[0], class2[1])
plt.scatter(class3[0], class3[1])

plt.legend(['Class 1','Class 2','Class 3'])
plt.title("Input Data")
plt.show()

def split_train_test(data):
  # Select ratio
  ratio = 0.70
  
  total_rows =data.shape[0]
  train_size = int(total_rows*ratio)
  
  # Split data into test and train
  train = data[0:train_size]
  test = data[train_size:]
  return train, test

training1, testing1 = split_train_test(class1)
training2, testing2 = split_train_test(class2)
training3, testing3 = split_train_test(class3)

tr1, tr2 = len(training1), len(training2)
t1, t2 = len(testing1), len(testing2)

training = training1.append(training2, ignore_index=True)
training = training.append(training3, ignore_index=True)
test = testing1.append(testing2, ignore_index=True)
test = test.append(testing3, ignore_index=True)

"""**Sigmoidal Function**"""

def sigmoidal(x):
  return 1/(1+np.exp(-(x)))

"""**model training function**"""

def train_fun(xn,y):
  p,q = xn.shape
  w = np.array([1,0,0]) #initializing weight
  er = []
  avg_er = []
  epoch = 100
   
  while (epoch):
    for i in range (p):
      xi = [1,xn.iloc[i,0], xn.iloc[i,1]]
      an = np.dot(w.T,xi)
      sn = sigmoidal(an)
      
      if i<y:
        yn = 0
      else:
        yn = 1
      
      error = 0.5*(yn-sn)**2
      lr = 0.3 #learning rate
      er.append(error) 

      d_w = lr*(yn-sn)*sn*(1-sn)
      d_w = np.dot(d_w, xi)
      w = w + d_w
    er_avg = sum(er)/len(er)
    avg_er.append(er_avg)
  
    #if er[i]-er[i-1]<10**(-3):
     # break
    epoch = epoch- 1
  print("new_weight: ",w)
  print("The Average Error: ", round(er_avg,6))
  print()
  return w, avg_er

"""**training the model**"""

tr_cl_12 = training1.append(training2)
w12, avg_er_12 = train_fun(tr_cl_12,len(training1))
#print(class12Tr)

tr_cl_13 = training1.append(training3)
w13, avg_er_13 = train_fun(tr_cl_13, len(training1))

tr_cl_23 = training2.append(training3)
w23, avg_er_23 = train_fun(tr_cl_23, len(training2))

"""**Testing the trained model**"""

def testing(xn, w):
    
    m, n = xn.shape
    yp = []
    
    for i in range(m):
        xi = [1, xn.iloc[i, 0], xn.iloc[i, 1]]
        an = np.dot(w.T, xi)
        yi = sigmoidal(an)
        yp.append(yi)

    return yp

"""**Plotting** **Decision Boundary between two class** """

def decision_boundary(all_points, data, w, name):
    
    c1x_, c2x_, c1y_, c2y_ = [], [], [], []
    c1, c2 = [], []
    
    yp = testing(all_points, w)
    m, n = all_points.shape
    for i in range(m):
        if yp[i] < 0.5:
            c1x_.append(all_points.iloc[i, 0])
            c1y_.append(all_points.iloc[i, 1])
            c1.append((all_points.iloc[i, 0],all_points.iloc[i, 1]))
        else:
            c2x_.append(all_points.iloc[i, 0])
            c2y_.append(all_points.iloc[i, 1])
            c2.append((all_points.iloc[i, 0],all_points.iloc[i, 1]))

    plt.scatter(c1x_, c1y_, marker='s', s=500, c= "orange")
    plt.scatter(c2x_, c2y_, marker='s', s=500, c= "red")
    plt.legend(['Class '+name[0],'Class '+name[1]])
    plt.scatter(data.iloc[:,:1], data.iloc[:,1:2], edgecolors='black')

    plt.title("Decision Boundary (Class"+name[0]+name[1]+")")
    plt.tight_layout()
   # plt.savefig(""+name[0]+name[1]+".png")
    plt.show()
    
    return [c1, c2]

max_x_val = int(max([max(class1.iloc[0]), max(class2.iloc[0]), max(class3.iloc[0])]))
min_x_val = int(min([min(class1.iloc[0]), min(class2.iloc[0]), min(class3.iloc[0])]))
max_y_val = int(max([max(class1.iloc[1]), max(class2.iloc[1]), max(class3.iloc[1])]))
min_y_val = int(min([min(class1.iloc[1]), min(class2.iloc[1]), min(class3.iloc[1])]))
all_points = []
for i in range(-15, 15):
    for j in range(-15,15):
        all_points.append([i, j])
        

all_points = pd.DataFrame(all_points)     
#all_points

class12T = testing1.append(testing2)
b12 = decision_boundary(all_points, class12T, w12, ['1', '2'])

class13T = testing1.append(testing3)
b13 = decision_boundary(all_points, class13T, w13, ['1', '3'])

class23T = testing2.append(testing3)
b23 = decision_boundary(all_points, class23T, w23, ['2', '3'])

"""**Merging all three decision boundries**"""

b1 = list(set.intersection(set(b12[0]), set(b13[0])))
b2 = list(set.intersection(set(b12[1]), set(b23[0])))
b3 = list(set.intersection(set(b13[1]), set(b23[1])))
b1x, b1y = [i[0] for i in b1], [i[1] for i in b1]
b2x, b2y = [i[0] for i in b2], [i[1] for i in b2]
b3x, b3y = [i[0] for i in b3], [i[1] for i in b3]

plt.scatter(b1x, b1y, marker='s', s=500, c="blue")
plt.scatter(b2x, b2y, marker='s', s=500, c="red")
plt.scatter(b3x, b3y, marker='s', s=500, c="magenta")


plt.scatter(class1[0], class1[1], edgecolors='black')
plt.scatter(class2[0], class2[1], edgecolors='black')
plt.scatter(class3[0], class3[1], edgecolors='black')
plt.legend(['Class 1','Class 2','Class 3'])
plt.title("Complete Decision Boundary")
plt.xlim(-15, 15)
plt.ylim(-15, 15)
#plt.savefig(" ")

plt.show()

print(avg_er_23)

"""**Error vs Epoch graph Plotting**"""

plt.plot(avg_er_12, color="orange")
plt.plot(avg_er_13, color="black")
plt.plot(avg_er_23, color="blue")
plt.legend(['Class 1','Class 2', 'Class 3'])
plt.title("Error vs Epoch")
plt.xlabel("Epoch")
plt.ylabel(" Error")
#plt.savefig(" ")
plt.show()

def confusion_matrix(predicted, k):
    matrix = 0
    predicted = convert_labels(predicted)
    c1, c2 = 0, 0
    for i in range(len(predicted)):
        if i<k and predicted[i] == 0:
            c1 += 1
        elif i>=k and predicted[i] == 1:
            c2 += 1
            
    return [[c1, k-c1], [k-c2, c2]]
def convert_labels(x):
    prediction = []
    for val in x:
        if val<0.5:
            prediction.append(0)
        else:
            prediction.append(1)

    return np.array(prediction)

#class12
pred12 = testing(class12T, w12)
matrix12 = confusion_matrix(pred12, len(testing1))
matrix12
print(len(pred12))

#class13
pred13 = testing(class13T, w13)
matrix13 = confusion_matrix(pred13, len(testing1))
matrix13
print(len(pred13))

#Class23
pred23 = testing(class23T, w23)
matrix23 = confusion_matrix(pred23, len(testing2))
matrix23
print(len(pred23))

#Overall Confusion Matrix
c11 = matrix12[0][0]+matrix13[0][0]
c12 = matrix12[0][1]
c13 = matrix13[0][1]

c21 = matrix12[1][0]
c22 = matrix12[1][1]+matrix23[0][0]
c23 = matrix23[0][1]

c31 = matrix13[1][0]
c32 = matrix23[1][0]
c33 = matrix13[1][1]+matrix23[1][1]

confusionmatrix = [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]]
print("Confusion Matrix")
print(np.array(confusionmatrix))

accuracy = (c11+c22+c33)/(2*(len(testing1)+len(testing2)+len(testing3)))*100
print('Accuracy = {}%'.format(accuracy))
print(len(testing3))

def function(x, y):
    return x/(x+y)
precision_1 = function(c11/2, c12+c13)
precision_2 = function(c22/2, c21+c23)
precision_3 = function(c33/2, c31+c32)
print('Precision\nClass1 = {}, Class2 = {}, Class3 = {}'.format(precision_1, precision_2, precision_3))

print("Average Precision: ", (precision_1+precision_2+precision_3)/3)

recall_1 = function(c11/2, c21+c31)
recall_2 = function(c22/2, c12+c32)
recall_3 = function(c33/2, c13+c23)
print('Recall\nClass1 = {}, Class2 = {}, Class3 = {}'.format(recall_1, recall_2, recall_3))

print("Average Recall: ", (recall_1+recall_2+recall_3)/3)

f_measure1 = (2*precision_1*recall_1)/(precision_1+recall_1)
f_measure2 = (2*precision_2*recall_2)/(precision_2+recall_2)
f_measure3 = (2*precision_3*recall_3)/(precision_1+recall_3)
print('F Score\nClass1 = {}, Class2 = {}, Class3 = {}'.format(f_measure1, f_measure2, f_measure3))

print("Average F-measure: ", (f_measure1+f_measure2+f_measure3)/3)
