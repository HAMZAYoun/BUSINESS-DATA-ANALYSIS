import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from google.colab import drive
drive.mount('/my-drive',force_remount=True)
path="/content/titanic.csv"
df=pd.read_csv(path)
df.describe
df=df[['Survived','Pclass','Sex','Age']]
df['Sex'].replace(['male','female'],[0,1],inplace=True)
x=df[['Sex','Pclass']]
y=df[['Survived']]
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x,y)
model.score(x,y)
def check(model,Sex=1,Pclass=2):
   xprime=np.array([Sex,Pclass]).reshape(1,2)
   print(model.predict(xprime))
   print(model.predict_proba(xprime))
    
    #Please feel free to update your neghbors number to find best estimate for your classification. 
