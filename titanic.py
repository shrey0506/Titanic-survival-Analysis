import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')
df_gender = pd.read_csv('gender_submission.csv')

df_train.info() #age and cabin contains null value
df_test.info()  #age and cabin contains null value
df_gender.info()

head_train = df_train.head()

figure, axis = plt.subplots(1,1,figsize=(18,4))
age = df_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data= age)
sns.factorplot(data= age)

figure = plt.figure(figsize=(15,6))
plt.hist([df_train[df_train['Survived']==1]['Age'].dropna(),df_train[df_train['Survived']==0]['Age'].dropna()], stacked=True, color = ['r','b'], bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Passengers Count')
plt.legend()

#Filling the NaN values in train data.(Data Munging)
narep = [df_train['Age'].median(),df_test['Age'].median()]
df_train['Age'].fillna(narep[0],inplace=True)
df_test['Age'].fillna(narep[0],inplace=True)

df_train.info()
df_test.info()

survived = df_train[df_train['Survived']==1]['Sex'].value_counts()
dead = df_train[df_train['Survived']==0]['Sex'].value_counts()
new_df = pd.DataFrame([survived,dead])
new_df.index =['survived','dead']
sns.pairplot(new_df)
new_df.plot(kind='bar',stacked=True, figsize=(10,6))


dataset = df_train.append(df_test)
dataset.info()

#categorizing field sex 
nds = dataset['Sex'].copy().values
nds[nds=='male']=1
nds[nds=='female']=0


grid = sns.FacetGrid(dataset, col ='Sex', size = 3.2, aspect =1.7)
grid.map(sns.barplot, 'Embarked','Survived', alpha= 0.6, ci = None)

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

Emb = dataset['Embarked'].copy().values
Emb[Emb=='S']=0
Emb[Emb=='C']=1
Emb[Emb=='Q']=2
Emb.shape



dataset.info()

dataset['Cabin'].fillna(dataset['Cabin'].mode()[0], inplace=True)
dataset.info()
dataset['Survived'].fillna(dataset['Survived'].mode()[0],inplace=True)
dataset.info()

dataset['Pclass'].head()
dataset.head()
dataset.fillna(dataset['Fare'].mode()[0], inplace=True)
dataset.info()

Cabin =dataset['Cabin'].copy().values
Cabin[Cabin=='C23']=0
Cabin[Cabin=='C25']=1
Cabin[Cabin=='C27']=2
Cabin.shape


new_dataset = dataset.copy()
new_dataset.loc[:,'Sex'] = nds #assigning numeric nd arrays to columns that held string vals.
new_dataset.loc[:,'Embarked'] = Emb
new_dataset.loc[:,'Cabin'] = Cabin
new_dataset.head(15) #after numerical feature transformation
 


pd.DataFrame(new_dataset)
new_dataset.drop('Name', axis=1, inplace=True)
new_dataset.drop('Cabin', axis=1, inplace=True)
new_dataset.drop('Ticket', axis=1, inplace=True)


new_dataset
y = new_dataset['Survived'].values
y.shape
y=y[:891]
x = new_dataset[:891]
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#SVM
from sklearn import svm

classifier1 =svm.SVC()
classifier1
classifier1.fit(x_train,y_train)
Y_pred = classifier1.predict(x_test)
acc_SVM = round(classifier1.score(x_train, y_train) * 100, 2)

#decesion Tree
from sklearn import tree

classifier2 = tree.DecisionTreeClassifier()
classifier2.fit(x_train,y_train)
Y_pred = classifier2.predict(x_test)
acc_decision_tree = round(classifier2.score(x_train, y_train) * 100, 2)

#logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(classifier2, x_train, y_train, cv=3)
confusion_matrix(y_train, predictions)

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))

from sklearn.metrics import f1_score
f1_score(y_train, predictions)
