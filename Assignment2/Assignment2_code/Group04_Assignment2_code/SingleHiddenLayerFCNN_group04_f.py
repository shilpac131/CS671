#!/usr/bin/env python
# coding: utf-8

# # FCNN Linear Data Classification

# In[160]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# In[161]:


file1path = "/home/ananya/Downloads/Group04/Classification/LS_Group04/Class1.txt"
file2path = "/home/ananya/Downloads/Group04/Classification/LS_Group04/Class2.txt"
file3path = "/home/ananya/Downloads/Group04/Classification/LS_Group04/Class3.txt"


# In[162]:


def readdata(file1path, file2path, file3path):
    
    class1 = pd.read_csv(file1path, header=None, delimiter=" ")
    class2 = pd.read_csv(file2path, header=None, delimiter=" ")
    class3 = pd.read_csv(file3path, header=None, delimiter=" ")
    
    return class1, class2, class3


# In[163]:


def split_data(data):
    m, n = data.shape
    l1 = int(0.6*m)
    l2 = l1 + int(0.2*m)
    training = data.iloc[:l1]
    validation = data.iloc[l1:l2]
    test = data.iloc[l2:m]
    
    return training, validation, test


# In[164]:


def plotplots(class1, class2, class3):
    
    plt.scatter(class1[0], class1[1])
    plt.scatter(class2[0], class2[1])
    plt.scatter(class3[0], class3[1])
    plt.legend(['Class 1','Class 2','Class 3'])
    plt.title("Input Data")
    plt.show()


# In[165]:


class1, class2, class3 = readdata(file1path, file2path, file3path)
plotplots(class1, class2, class3)

training1, validation1, test1 = split_data(class1)
training2, validation2, test2 = split_data(class2)
training3, validation3, test3 = split_data(class3)

tr1, tr2, tr3 = len(training1), len(training2), len(training3)
v1, v2, v3 = len(validation1), len(validation2), len(validation3)
t1, t2, t3 = len(test1), len(test2), len(test3)


# In[166]:


training = training1.append(training2, ignore_index=True)
training = training.append(training3, ignore_index=True)

validation = validation1.append(validation2, ignore_index=True)
validation = validation.append(validation3, ignore_index=True)

test = test1.append(test2, ignore_index=True)
test = test.append(test3, ignore_index=True)


# # FCNN Model

# In[167]:


def sigmoid(x):
    return 1/(1+np.exp(-(x)))

def inst_error(y, yp):
    return sum(0.5*(y-yp)**2)


# In[168]:


def train_data(xn, l1, l2):

    wh = 0.10*np.random.randn(3, 4)
    wo = 0.10*np.random.randn(4, 3)
    avg_err = []
    epoch = 100
    
    while(epoch):
    
        err = []
        c1, c2, c3 = 0, 0, 0
        
        for i in range(len(xn)):

            xi = [1, xn.iat[i,0], xn.iat[i,1]] #Input layer
            #--------------------------
            h1 = np.dot(wh.T, xi) #hidden layer 1 out
            a1 = sigmoid(h1) #hidden layer 1 activation out
            #--------------------------
            out = np.dot(wo.T, a1) #Output layer out
            ao = sigmoid(out) #Output activation 
            #--------------------------

            if i<l1:
                y = [1, 0, 0]
                en = inst_error(y, ao)
                c1+=1
            elif i>=l1 and i<(l1+l2):
                y = [0, 1, 0]
                en = inst_error(y, ao)
                c2+=1
            else:
                y = [0, 0, 1]
                en = inst_error(y, ao)
                c3+=1
                
                
            err.append(en)
            neta = 0.01
            
            #Update weights
            wo = wo + (neta * np.outer(a1, ((y-ao) * ao * (1-ao))))
            
            var1 = np.dot(wo, ((y-ao) * ao * (1-ao)))
            var2 = a1 * (1-a1)
            
            wh = wh + (neta * np.outer(xi, var1*var2))

            
        avg_error = sum(err)/len(err)
        avg_err.append(avg_error)
        if epoch%5==0:
            print("Error: ",avg_error)
        epoch -= 1
    
    print(c1, c2, c3)
    return wh, wo, avg_err, a1, ao

wh, wo, avg_err, h1_out, o_out = train_data(training, tr1, tr2)


# In[169]:


def test_data(wh, wo, xn):
    pred_out = []
    neuron=[]
    for i in range(len(xn)):
        xi = [1, xn.iat[i,0], xn.iat[i,1]]
        h1 = np.dot(wh.T, xi)
        a1 = sigmoid(h1)
        neuron.append(a1)
        out = np.dot(wo.T, a1) 
        ao = sigmoid(out)
        
        pred_out.append(ao)
        
    return np.round(np.array(pred_out)),np.round(np.array(neuron))


# In[170]:


max_x_val = int(max([max(class1.iloc[0]), max(class2.iloc[0]), max(class3.iloc[0])]))
min_x_val = int(min([min(class1.iloc[0]), min(class2.iloc[0]), min(class3.iloc[0])]))
max_y_val = int(max([max(class1.iloc[1]), max(class2.iloc[1]), max(class3.iloc[1])]))
min_y_val = int(min([min(class1.iloc[1]), min(class2.iloc[1]), min(class3.iloc[1])]))


# In[171]:


all_points = []
for i in range(min_x_val-15, max_x_val+15):
    for j in range(min_y_val-15, max_y_val+15):
        all_points.append([i, j])
        

all_points = pd.DataFrame(all_points)     
#all_points


# In[172]:


output,neuron = test_data(wh, wo, all_points)


# In[173]:


c1x, c2x, c3x = [], [], []
c1y, c2y, c3y = [], [], []

y = output

for i in range(len(all_points)):
        
    if y[i][0] == 1:
        c1x.append(all_points.iloc[i, 0])
        c1y.append(all_points.iloc[i, 1])
    elif y[i][1] == 1:
        c2x.append(all_points.iloc[i, 0])
        c2y.append(all_points.iloc[i, 1])
    else:
        c3x.append(all_points.iloc[i, 0])
        c3y.append(all_points.iloc[i, 1])
    

plt.scatter(c1x, c1y,marker='s',s=150)
plt.scatter(c2x, c2y,marker='s' ,s=150)
plt.scatter(c3x, c3y, marker='s', s=150)




plt.legend(['Class 1','Class 2', 'Class 3'])
plt.legend(['Class 1','Class 2', 'Class 3'])


plt.scatter(class1[0], class1[1], c='blue',edgecolors='black')
plt.scatter(class2[0], class2[1], c='orange',edgecolors='black')
plt.scatter(class3[0], class3[1], c='green',edgecolors='black')

plt.title("Decision Boundary")
plt.tight_layout()

plt.xlim(-12, 21)
plt.ylim(-20, 12)

plt.show()


# In[174]:


plt.plot(avg_err)
plt.title("Average Error vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.show()

# In[175]:


def confusion_matrix(yp, l1, l2):
    matrix = 0
    c11, c12, c13, c21, c22, c23, c31, c32, c33 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(yp)):
        if i<l1:
            if yp[i][0] == 1:
                c11 += 1
            elif yp[i][1] == 1:
                c12 += 1
            else:
                c13 += 1
        elif i>=l1 and i< (l1+l2):
            if yp[i][1] == 1:
                c22 += 1
            elif yp[i][0] == 1:
                c21 += 1
            else:
                c23 += 1
        else:
            if yp[i][2] == 1:
                c33 += 1
            elif yp[i][0] == 1:
                c31 += 1
            else:
                c32 += 1
    
            
    return [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]]


# In[176]:

print('Confusion matrix for test data:')
yp,neuron_t = test_data(wh, wo, test)
matrix = confusion_matrix(yp, len(test1), len(test2))
[c11, c12, c13], [c21, c22, c23], [c31, c32, c33] = matrix
print(np.array(matrix))


# In[177]:


accuracy = (c11+c22+c33)/((len(test1)+len(test2)+len(test3)))*100
print('Accuracy = {}%'.format(accuracy))


# In[178]:


def function(x, y):
    if((x+y)==0):
        return 0.0
    else:
        return x/(x+y)

def function1(x, y):
    if((x+y)==0):
        return 0.0
    else:
        return (2*x*y)/(x+y)


# In[179]:


precision_1 = function(c11/200, c12+c13)
precision_2 = function(c22/200, c21+c23)
precision_3 = function(c33/200, c31+c32)
print('Precision\nClass1 = {}, Class2 = {}, Class3 = {}'.format(precision_1, precision_2, precision_3))

print("Average Precision: ", (precision_1+precision_2+precision_3)/3)


# In[180]:


recall_1 = function(c11/200, c21+c31)
recall_2 = function(c22/200, c12+c32)
recall_3 = function(c33/200, c13+c23)
print('Recall\nClass1 = {}, Class2 = {}, Class3 = {}'.format(precision_1, precision_2, precision_3))

print("Average Recall: ", (recall_1+recall_2+recall_3)/3)


# In[181]:


f_measure1 = function1(precision_1,recall_1)
f_measure2 = function1(precision_2,recall_2)
f_measure3 = function1(precision_3,recall_3)
print('F Score\nClass1 = {}, Class2 = {}, Class3 = {}'.format(f_measure1, f_measure2, f_measure3))

print("Average F-measure: ", (f_measure1+f_measure2+f_measure3)/3)


# In[202]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(test.iloc[:,0], test.iloc[:,1],  neuron_t[:,0], c=neuron_t[:,0], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on testing data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 1 output")

plt.show()


# In[203]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(test.iloc[:,0], test.iloc[:,1],  neuron_t[:,1], c=neuron_t[:,1], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on testing data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 2 output")

plt.show()


# In[204]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(test.iloc[:,0], test.iloc[:,1],  neuron_t[:,2], c=neuron_t[:,2], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on testing data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 3 output")

plt.show()


# In[206]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(test.iloc[:,0], test.iloc[:,1],  neuron_t[:,3], c=neuron_t[:,3], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on testing data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 4 output")

plt.show()


# In[1]:

print('Confusion matrix for training data')
yp1,neuron_t1 = test_data(wh, wo, training)
matrix = confusion_matrix(yp, len(test1), len(test2))
[c11, c12, c13], [c21, c22, c23], [c31, c32, c33] = matrix
print(np.array(matrix))


# In[189]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(training.iloc[:,0], training.iloc[:,1],  neuron_t1[:,0], c=neuron_t1[:,0], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on training data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 1 output")

plt.show()


# In[190]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(training.iloc[:,0], training.iloc[:,1],  neuron_t1[:,1], c=neuron_t1[:,1], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on training data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 2 output")

plt.show()


# In[191]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(training.iloc[:,0], training.iloc[:,1],  neuron_t1[:,2], c=neuron_t1[:,2], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on training data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 3 output")

plt.show()


# In[192]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(training.iloc[:,0], training.iloc[:,1],  neuron_t1[:,3], c=neuron_t1[:,3], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on training data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 4 output")

plt.show()


# In[194]:

print('Confusion matrix for validation data')
yp2,neuron_t2 = test_data(wh, wo, validation)
matrix = confusion_matrix(yp, len(test1), len(test2))
[c11, c12, c13], [c21, c22, c23], [c31, c32, c33] = matrix
print(np.array(matrix))


# In[198]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(validation.iloc[:,0], validation.iloc[:,1],  neuron_t2[:,0], c=neuron_t2[:,0], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on validation data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 1 output")

plt.show()


# In[199]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(validation.iloc[:,0], validation.iloc[:,1],  neuron_t2[:,1], c=neuron_t2[:,1], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on validation data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 2 output")

plt.show()


# In[200]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(validation.iloc[:,0], validation.iloc[:,1],  neuron_t2[:,2], c=neuron_t2[:,2], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on validation data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 3 output")

plt.show()


# In[201]:


fig = plt.figure(figsize = (15,4))
ax = plt.axes(projection='3d')
scat = ax.scatter(validation.iloc[:,0], validation.iloc[:,1],  neuron_t2[:,3], c=neuron_t2[:,3], cmap="winter")
plt.colorbar(scat)
#ax.scatter3D(testing.iloc[:,0], testing.iloc[:,1], neuron_t[:,0], color='green')
plt.title("Output of Hidden node on validation data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
ax.set_zlabel("Neuron 4 output")

plt.show()


# In[ ]:




