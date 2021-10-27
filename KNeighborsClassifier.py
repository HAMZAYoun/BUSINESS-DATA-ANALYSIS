import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
def check(model,Sex=1,Pclass=2):                    #feel free to plug any estimated features, and enjoy watching the result         
   xprime=np.array([Sex,Pclass]).reshape(1,2)
   print(model.predict(xprime))
   print(model.predict_proba(xprime))
check(model)                                        #feel free to check both prediction and probability result
    
#Feel free to upload a titanic dataset from the internet and work on adding it 
#on code lab, syntax for upload is updated. if you feel working on pycharm 
# ignore code lines between 6-8
