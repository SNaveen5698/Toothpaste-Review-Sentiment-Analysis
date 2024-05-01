import pandas as pd 
df=pd.read_csv("/content/labelled colgate product reviews.csv")
df.head()
df.isnull().sum()
df.drop(["Unnamed: 2","Unnamed: 3"],axis=1,inplace=True)
x=df["Text"]
y=df["Label"]
from sklearn.feature_extraction.text import CountVectorizer
vector=CountVectorizer()
x=vector.fit_transform(x)
from sklearn.model_selection import train_test_split 
a,b,c,d=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
obj=LogisticRegression()
obj.fit(a,c)
ycap=obj.predict(b)
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix 
print(accuracy_score(d,ycap))
new=["colgate paste is good it is the best for teeth health"]
new=vector.transform(new)

print(obj.predict(new)) 