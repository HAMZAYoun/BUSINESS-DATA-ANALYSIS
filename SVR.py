
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.svm import SVR
num=100
outliers=5
xprime2=np.random.normal(size=200, scale=0.25)
xprime1=np.random.rand(outliers)
x = np.concatenate([xprime1,xprime2]).reshape(205,1)
yprime1= np.random.rand(outliers)
yprime2= xprime2*xprime2
y = np.concatenate([yprime1,yprime2])
plt.scatter(x,y,color='green')
model = SVR(C=100)
model.fit(x,y)
model.score(x,y)
p=model.predict(x)
plt.scatter(x,y,color='g')
plt.plot(x,p,color='r')
plt.title('SVR Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['SVR_model','Data_Set'],loc='best')
plt.rcParams['figure.figsize'] = (40.0, 5.0)
