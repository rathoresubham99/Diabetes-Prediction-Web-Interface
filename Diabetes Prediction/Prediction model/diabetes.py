
# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle


# In[5]:


data = pd.read_csv("diabetes-pima.csv")


# In[6]:


data.head(10)


# In[7]:


# to check if any null value is present
data.isnull().values.any()


# In[8]:


## checking Correlation
# to get correlations of each features in dataset
correlation_matrix = data.corr()
corr_features = correlation_matrix.index
plt.figure(figsize=(10,5))
#plottin the  heat map
g=sns.heatmap(data[corr_features].corr(),annot=True)


# In[9]:


data.corr()


# In[12]:


data.head(5)


# In[13]:


diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])


# In[14]:



(diabetes_true_count,diabetes_false_count)


# In[17]:


dataX = data.iloc[:,[1, 4, 5, 7]].values    # since the correlation of BMI,Age, Insulin and Diabetic Conc. has higher correlation with the outcome
dataY = data.iloc[:,8].values     


# In[18]:


dataX


# In[19]:


dataY


# In[20]:



from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataX)


# In[21]:



dataset_scaled = pd.DataFrame(dataset_scaled)


# In[22]:


X = dataset_scaled


# In[23]:


X


# In[24]:


Y=dataY


# In[25]:



Y


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 45, stratify = data['Outcome'] )


# In[27]:

from sklearn.impute import SimpleImputer

fill_values =SimpleImputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[28]:


X_train


# In[29]:


X_test


# In[30]:


import xgboost


# In[31]:


classifier=xgboost.XGBClassifier()


# In[32]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.04, max_delta_step=0, max_depth=3,
              min_child_weight=5, missing=0, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)


# In[40]:


classifier.fit(X_train,Y_train)


# In[41]:


y_pred=classifier.predict(X_test)


# In[42]:


y_pred


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[44]:


cm=confusion_matrix(Y_test,y_pred)
score=accuracy_score(Y_test,y_pred)


# In[45]:


cm


# In[46]:


score


# In[48]:



pickle.dump(classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




