

# In[54]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
## 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score


# In[55]:


df = pd.read_csv('dataset-manual.csv')
df.head()


# In[56]:


df.drop(['Symptom_8','Symptom_9','Symptom_10','Symptom_11','Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17'],axis=1,inplace=True)


# In[57]:


df.shape


# In[58]:


# Percentage of Missing Values
(df.isnull().sum()/df.shape[0])*100


# In[59]:


df.columns


# In[60]:


df.describe().T


# In[61]:

for row in list(df.iterrows()):
    row[1][1:].sort_values(inplace=True)

#Remove Hyphen
for col in df.columns:
    df[col]= df[col].str.replace('_',' ')


# In[62]:


cols = df.columns

data = df[cols].values.flatten()

reshaped = pd.Series(data)
reshaped = reshaped.str.strip()
reshaped = reshaped.values.reshape(df.shape)

df = pd.DataFrame(reshaped, columns = df.columns)
df.head()


# In[63]:


df.fillna(0,inplace=True)
df.head()


# In[64]:


df.isna().sum()


# In[65]:


df['Disease'].value_counts()


# In[66]:


#Importing Symptoms Dataset

df_s = pd.read_csv('severity.csv')
df_s.head()


# In[67]:


# Remove Hyphen
df_s['Symptom']=df_s['Symptom'].str.replace('_',' ')
df_s['Symptom'].unique()


# In[68]:


a= np.array(df_s['weight'])



# In[69]:


#Encoding Symptoms

vals = df.values
symptoms = df_s['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df_s[df_s['Symptom'] == symptoms[i]]['weight'].values[0]
    
newdf = pd.DataFrame(vals, columns=cols)
newdf.head()


# In[70]:


#No Symptoms assiging zero
'''newdf = newdf.replace('dischromic  patches', 0)
newdf = newdf.replace('spotting  urination',0)
newdf = newdf.replace('foul smell of urine',0)
newdf.head(10)
'''

# In[71]:


#Selection of features for Training Purpose
X = newdf.drop(['Disease'],axis=1)
y = newdf['Disease']
X.head()


# In[72]:


y.sample(6)


# In[73]:


#Splitting the dataset
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score
from sklearn.svm import SVC


# In[74]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify=y,random_state=0)

# In[76]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators=700,random_state=0,n_jobs=-1,verbose=4)
clf_rfc.fit(X_train,y_train)
predict = clf_rfc.predict(X_test)
print('Accuracy Score: {}%'.format(round(accuracy_score(y_test,predict)*100,2)))
print(classification_report(y_test,predict))


# In[77]:


#Function for prediction

def predict(s1,s2,s3,s4='Nil',s5='Nil',s6='Nil',s7='Nil'):
    l = [s1,s2,s3,s4,s5,s6,s7]
    print(l)
    
    x= np.array(df_s['Symptom'])
    y= np.array(df_s['weight'])
    for i in range(len(l)):
        for j in range(len(x)):
            if l[i]==x[j]:
                l[i]=y[j]
    res = [l]
    pred = clf_rfc.predict(res)
    print(pred[0])
    return pred


# In[78]:


m=predict('fever','Red spots over body','Nil')
p=m[0]

# In[79]:


# saving model
import pickle
pickle.dump(clf_rfc,open('model.pkl','wb'))


# In[ ]:

med=pd.read_csv('medicines.csv')
x1=np.array(med['Disease'])
y1=np.array(med['Medicine'])
for i in range(len(x1)):
    if p==x1[i]:
        print("You have been detected with: ",p)
        print("Your medicine is: ",y1[i])
        print("Please consult a doctor if the symptoms persist")





def performance_evaluator(model, X_test, y_test):
    """
    model: Load the trained model
    X_test: test data
    y_test: Actual value
    
    """
    
    y_predicted = model.predict(X_test)
    
    precision = precision_score(y_test, y_predicted,average='micro')*100
    
    accuracy = accuracy_score(y_test, y_predicted)*100
    
    f1 = f1_score(y_test, y_predicted, average='macro')*100
    
    recall = recall_score(y_test, y_predicted, average='macro')*100
    
    print('precision----->', precision) 
    print('\n************************')
    print('Accuracy----->', accuracy)
    print('\n************************')
    print('F1 Score----->', f1)
    print('\n************************')
    print('Recall----->', recall)
    print('\n************************')
    return accuracy, precision, f1, recall



# In[64]:


## support Vector machine Hyperparameter tuned

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
grid.fit(X_train, y_train)


# In[67]:


## best estimator

print(grid.best_estimator_)
print(grid.best_params_)


# In[68]:


## lets built based SVC model.

hyper_tuned_svc = SVC(C= 10, gamma= 0.1, kernel= 'rbf')
hyper_tuned_svc.fit(X_train, y_train)

## lets calculate performance
_1, _2, _3, _4 = performance_evaluator(hyper_tuned_svc, X_test, y_test)




# In[ ]:




