# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:42:27 2020

@author: Prasanth
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('F:\ASS\ENGLISH')

df.columns

x=df.copy()
del x['letter (Target)']

y=df['letter (Target)']

plt.boxplot(x['xbox'])
plt.boxplot(x['width'])
plt.boxplot(x['height'])
plt.boxplot(x['onpix'])
plt.boxplot(x['xbar'])
plt.boxplot(x['ybar'])
plt.boxplot(x['x2bar']) 
plt.boxplot(x['y2bar']) 
plt.boxplot(x['xybar']) 
plt.boxplot(x['x2ybar']) 
plt.boxplot(x['xy2bar']) 
plt.boxplot(x['xedge']) 
plt.boxplot(x['xedgey']) 
plt.boxplot(x['yedge'])
plt.boxplot(x['yedgex'])

x.isnull().sum()

per=x['xbox'].quantile([0,0.97]).values
x['xbox']=x['xbox'].clip(per[0],per[1])

per=x['width'].quantile([0.1,0.97]).values
x['width']=x['width'].clip(per[0],per[1])

per=x['height'].quantile([0,0.98]).values
x['height']=x['height'].clip(per[0],per[1])

per=x['onpix'].quantile([0,0.98]).values
x['onpix']=x['onpix'].clip(per[0],per[1])

per=x['xbar'].quantile([0.05,0.98]).values
x['xbar']=x['xbar'].clip(per[0],per[1])

per=x['ybar'].quantile([0.05,0.978]).values
x['ybar']=x['ybar'].clip(per[0],per[1])


per=x['x2bar'].quantile([0,0.962]).values
x['x2bar']=x['x2bar'].clip(per[0],per[1])

per=x['y2bar'].quantile([0,0.99]).values
x['y2bar']=x['y2bar'].clip(per[0],per[1])


per=x['xybar'].quantile([0.05,0.98]).values
x['xybar']=x['xybar'].clip(per[0],per[1])


per=x['x2ybar'].quantile([0.05,0.975]).values
x['x2ybar']=x['x2ybar'].clip(per[0],per[1])


per=x['xy2bar'].quantile([0.1,0.96]).values
x['xy2bar']=x['xy2bar'].clip(per[0],per[1])


per=x['xedge'].quantile([0,0.962]).values
x['xedge']=x['xedge'].clip(per[0],per[1])


per=x['xedgey'].quantile([0.15,0.9]).values
x['xedgey']=x['xedgey'].clip(per[0],per[1])



per=x['yedge'].quantile([0,0.976]).values
x['yedge']=x['yedge'].clip(per[0],per[1])


per=x['yedgex'].quantile([0.07,0.976]).values
x['yedgex']=x['yedgex'].clip(per[0],per[1])


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=90)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
sd.fit(xtrain,ytrain)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(xtrain,ytrain)

ypred_rts=reg.predict(xtest)
ypred_rtr=reg.predict(xtrain)

reg.coef_
reg.score

from sklearn.metrics import accuracy_score
accuracy_score(ytrain,ypred_rtr)#71
accuracy_score(ytest,ypred_rts)#71

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred_rts))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred_rts))

















































































































