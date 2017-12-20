# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:58:42 2017

@author: yangsu
"""
#reference
#https://m.blog.naver.com/samsjang/220967436415
#only for irisi example

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap


style.use('seaborn-talk')


krfont={'family': 'NanumGothic', 'weight':'bold', 'size':10}
matplotlib.rc('font', **krfont)
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_decision_region(x,y,classifier, test_idx=None, resolution=0.2, title=''):
    markers = ('s', 'x', 'o', '^', 'v') #shape of dots
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
     # x = np.array([10, 10, 20, 20, 30, 30])
     #(np.unique(x))
     #[10 20 30] 
     
     #draw decision surface
    x1_min, x1_max = x[:,0].min()-1, x[:,0].max()+1
    x2_min, x2_max = x[:,1].min()-1, x[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
     # draw grid x1.min~x1.max, x2.min~x2.max

    z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
     #ravel() : made 1-d array/ array().T : transpose array
    z=z.reshape(xx1.shape)
     #restore to original shape
     
    plt.contourf(xx1,xx2,z,alpha=0.5, cmap=cmap)
     #draw contourf(등고선)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
     

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)
         
    if test_idx:
        plt.scatter(x_test[:,0], x_test[:,1], c='', linewidth=1, marker='o', s=80, label='testSet')
         
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

if __name__=='__main__':
    iris=datasets.load_iris()
    x=iris.data[:,[2,3]]
    y=iris.target
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
    
    sc = StandardScaler()
    sc.fit(x_train) # calcuate average and Standard Deviation of x_train
    x_train_std=sc.transform(x_train) # regulation x_train
    x_test_std=sc.transform(x_test) # regulation x_test
    
    
    ml=SVC(kernel='sigmoid', C=10.0, gamma=0.10, random_state=0)
    ml.fit(x_train_std, y_train)
    y_pred = ml.predict(x_test_std)
    
    print('total test set : %d, total error : %d' %(len(y_test), (y_test != y_pred).sum()))
    print('accuracy : %.2f' %accuracy_score(y_test, y_pred))
    
    x_total = np.vstack((x_train_std, x_test_std)) #stack vertical
    y_total = np.hstack((y_train, y_test)) # stack horizental
    plot_decision_region(x=x_total, y=y_total, classifier=ml, title='scikit-learn SVM RBF')