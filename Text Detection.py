# Fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist=fetch_openml('mnist_784')
x,y=mnist['data'],mnist['target']

#Error
some_digit=x[3601]
some_digit_image=some_digit.reshape(28,28) #Lets reshape it to plot it
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis('off')

x_train,x_test=x[:6000],x[6000:7000]
y_train,y_test=y[:6000],y[6000:7000]

import numpy as np
#random shuffling
shuffle_index=np.random.permutation(6000)
x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]

#creating a 2 detector
y_train=y_train.astype(np.int8) #converts str into int
y_test=y_test.astype(np.int8) #converts str into int
y_train_2=(y_train==2) #Gives true / false
y_test_2=(y_test==2) #Gives true / false

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(tol=0.1) #adding tolerance to improve speed
clf.fit(x_train,y_train_2)
clf.predict([some_digit])

# Doing cross validation
from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train,cv=3,scoring='accuracy')
print(a.mean()) #gives accuracy