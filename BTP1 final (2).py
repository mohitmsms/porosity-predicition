#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[99]:


import tensorflow


# In[102]:


df=pd.read_excel("TOTALDATA REAL.xlsx")
df


# In[103]:


df.columns


# In[104]:


df0=df.drop([' ', ' .1','BEST THERMAL POROSITY','IMAGE DERIVED DENSITY'],axis=1)


# In[105]:


df1=df0.replace(-999.25,np.NaN)
df1


# In[106]:


df2=df1.dropna()
df2


# In[107]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(copy=True)


# In[108]:


scaler.fit(df2)


# In[109]:


df3=pd.DataFrame(scaler.transform(df2),index=df2.index, columns=df2.columns)
df3


# In[110]:


#import base64
#import pandas as pd
#from IPython.display import HTML

#def create_download_link( df3, title = "Download CSV file", filename = "data.csv"):
 #   csv = df3.to_csv()
  #  b64 = base64.b64encode(csv.encode())
   # payload = b64.decode()
 #   html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
  #  html = html.format(payload=payload,title=title,filename=filename)
   # return HTML(html)

#df3 = pd.DataFrame(scaler.transform(df3),index=df3.index, columns=df3.columns)
#create_download_link(df)


# In[111]:


#df3.to_csv(r'C:\Users\Mohit\Desktop.csv')


# In[112]:


Y=df3["TOTAL POROSITY"]
Y


# In[113]:


X=df3.drop("TOTAL POROSITY",axis=1)
X


# In[114]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)


# In[115]:


X.value_counts()


# In[116]:


Y.value_counts()


# In[117]:


X_train.shape


# In[118]:


Y_train.shape


# In[119]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()


# In[120]:


model.fit(X_train,Y_train)


# In[121]:


random_pred= model.predict(X_test)


# In[122]:


from sklearn.metrics import r2_score


# In[123]:


r2= r2_score(Y_test,random_pred)
r2


# In[124]:


from sklearn.metrics import mean_squared_error


# In[125]:


randomforest_rmse=mean_squared_error(Y_test,random_pred,squared=False)
randomforest_rmse


# In[126]:


model.score(X_train,Y_train)


# In[127]:


model.score(X_test,Y_test)


# In[128]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(Y_test,random_pred))
print(accuracy_score(Y_test,random_pred))
#print(classification_report(Y_test,random_pred))


# In[129]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats


# In[130]:


#Random Forest
clf = RandomForestRegressor()
scores = cross_val_score(clf, X, Y, cv=3,scoring='neg_mean_squared_error') # 3-fold cross-validation
print("MSE:"+ str(-scores.mean()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[131]:


from sklearn.linear_model import LinearRegression
model1=LinearRegression()


# In[132]:


model1.fit(X_train,Y_train)


# In[133]:


model1.fit(X_train,Y_train)


# In[134]:


model1.score(X_train,Y_train)


# In[135]:


model1.score(X_test,Y_test)


# In[136]:


lr_pred= model1.predict(X_test)
r2_lr= r2_score(Y_test,lr_pred)


# In[137]:


r2_lr


# In[138]:


lr_rmse=mean_squared_error(Y_test,lr_pred,squared=False)
lr_rmse


# In[139]:


from sklearn import svm


# In[140]:


model2=svm.SVR()


# In[141]:


model2.fit(X_train,Y_train)


# In[142]:


svm_pred= model2.predict(X_test)


# In[143]:


svm_r2= r2_score(Y_test,svm_pred)
svm_r2


# In[144]:


svm_rmse=mean_squared_error(Y_test,svm_pred,squared=False)
svm_rmse


# In[145]:


model2.score(X_test,Y_test)


# In[146]:


#SVM
clf = SVR(gamma='scale')
scores = cross_val_score(clf, X, Y, cv=3,scoring='neg_mean_squared_error')
print("MSE:"+ str(-scores.mean()))


# In[147]:


from sklearn import linear_model
model4=linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)


# In[148]:


model4.fit(X_train,Y_train)


# In[149]:


model4.score(X_test,Y_test)


# In[150]:


model4.score(X_train,Y_train)


# In[151]:


#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.svm import SVR
#from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

#models = []
#models.append(('KNN', KNeighborsRegressor(n_neighbors = 5)))
#models.append(('DT', DecisionTreeRegressor()))
#models.append(('SVM', SVR(kernel='linear')))
#models.append(('GBR', GradientBoostingRegressor()))
#models.append(('RF', RandomForestRegressor(n_estimators=250)))


# In[152]:


#from sklearn import model_selection

#accuracy = []
#model_name = []

#for name, model in models:
 #   kfold = model_selection.KFold(n_splits=10)
  #  cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold)
   # accuracy.append(cv_results)
    #model_name.append(name)


# In[153]:


#from sklearn.pipeline import make_pipeline
#clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
#cross_val_score(clf, X, Y, cv=cv)


# In[154]:


#plt.boxplot(accuracy)
#plt.title('Box Plot for Model Accuracy of CST(T=0) Predictions')
#plt.xlabel('Regression Model')
#plt.ylabel('Model Accuracy')
#plt.xticks([1,2,3,4,5,6], labels=model_name)
#plt.show()


# In[155]:


from sklearn import tree
model5=tree.DecisionTreeRegressor()


# In[156]:


model5.fit(X_train,Y_train)


# In[157]:


model5.score(X_test,Y_test)


# In[158]:


model5.score(X_train,Y_train)


# In[159]:


DT_pred= model5.predict(X_test)
r2_DT= r2_score(Y_test,lr_pred)


# In[160]:


r2_DT


# In[161]:


DT_rmse=mean_squared_error(Y_test,DT_pred,squared=False)
DT_rmse


# In[162]:


from sklearn import linear_model
model6 = linear_model.BayesianRidge()


# In[163]:


model6.fit(X_train,Y_train)


# In[164]:


model6.score(X_test,Y_test)


# In[165]:


model6.score(X_train,Y_train)


# In[166]:


BR_pred= model6.predict(X_test)
r2_BR= r2_score(Y_test,lr_pred)


# In[167]:


r2_BR


# In[168]:


BR_rmse=mean_squared_error(Y_test,BR_pred,squared=False)
BR_rmse


# In[169]:


from sklearn.neural_network import MLPClassifier


# In[170]:


model7 = MLPClassifier(solver='lbfgs', alpha=1e-5,
               hidden_layer_sizes=(5, 2), random_state=1)


# In[171]:


#model7.fit(X_train,Y_train)


# In[172]:


#model7.fit(X, Y)
#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
         #     solver='lbfgs')


# In[173]:


from sklearn.preprocessing import StandardScaler  
model8 = StandardScaler()  


# In[180]:


#ANN
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
def ANN(optimizer = 'adam',neurons=32,batch_size=32,epochs=50,activation='relu',patience=5,loss='mse'):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X.shape[1],), activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience
    history = model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              callbacks = [early_stopping],
              verbose=0) #verbose set to 1 will show the training process
    return model


# In[181]:


clf = KerasRegressor(build_fn=ANN, verbose=0)
scores = cross_val_score(clf, X, Y, cv=3,scoring='neg_mean_squared_error')
print("MSE:"+ str(-scores.mean()))


# In[182]:


#Random Forest
from sklearn.model_selection import GridSearchCV
# Define the hyperparameter configuration space
rf_params = {
    'n_estimators': [10, 20, 30],
    #'max_features': ['sqrt',0.5],
    'max_depth': [15,20,30,50],
    #'min_samples_leaf': [1,2,4,8],
    #"bootstrap":[True,False],
    #"criterion":['mse','mae']
}
clf = RandomForestRegressor(random_state=0)
grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
grid.fit(X, Y)
print(grid.best_params_)
print("MSE:"+ str(-grid.best_score_))


# In[183]:


#SVM
from sklearn.model_selection import GridSearchCV
rf_params = {
    'C': [1,10, 100],
    "kernel":['poly','rbf','sigmoid'],
    "epsilon":[0.01,0.1,1]
}
clf = SVR(gamma='scale')
grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
grid.fit(X, Y)
print(grid.best_params_)
print("MSE:"+ str(-grid.best_score_))


# In[184]:


#KNN
from sklearn.model_selection import GridSearchCV
rf_params = {
    'n_neighbors': [2, 3, 5,7,10]
}
clf = KNeighborsRegressor()
grid = GridSearchCV(clf, rf_params, cv=3, scoring='neg_mean_squared_error')
grid.fit(X, Y)
print(grid.best_params_)
print("MSE:"+ str(-grid.best_score_))


# In[185]:


#Random Search


# In[186]:


#Random Forest
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
# Define the hyperparameter configuration space
rf_params = {
    'n_estimators': sp_randint(10,100),
    "max_features":sp_randint(1,13),
    'max_depth': sp_randint(5,50),
    "min_samples_split":sp_randint(2,11),
    "min_samples_leaf":sp_randint(1,11),
    "criterion":['mse','mae']
}
n_iter_search=20 #number of iterations is set to 20, you can increase this number if time permits
clf = RandomForestRegressor(random_state=0)
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, Y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))


# In[187]:


#SVM
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    'C': stats.uniform(0,50),
    "kernel":['poly','rbf','sigmoid'],
    "epsilon":stats.uniform(0,1)
}
n_iter_search=20
clf = SVR(gamma='scale')
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, Y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))


# In[188]:


#KNN
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    'n_neighbors': sp_randint(1,20),
}
n_iter_search=10
clf = KNeighborsRegressor()
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, Y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))


# In[189]:


#ANN
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    'optimizer': ['adam','rmsprop'],
    'activation': ['relu','tanh'],
    'loss': ['mse','mae'],
    'batch_size': [16,32,64],
    'neurons':sp_randint(10,100),
    'epochs':[20,50],
    #'epochs':[20,50,100,200],
    'patience':sp_randint(3,20)
}
n_iter_search=10
clf = KerasRegressor(build_fn=ANN, verbose=0)
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, Y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


model8.fit(X_train)


# In[75]:


X_train = model8.transform(X_train)


# In[76]:


X_test = model8.transform(X_test) 


# In[77]:


from matplotlib import pyplot as plt


# In[78]:


import seaborn as sns


# In[79]:


df2['TOTAL POROSITY'].nunique()


# In[80]:


sns.displot(x='TOTAL POROSITY',data=df2)


# In[81]:


sns.boxplot(df2['TOTAL POROSITY'])


# In[82]:


histdf = df2.hist(bins=100,figsize=(15,10))


# In[83]:


sns.lineplot(data=df2['IMAGE DERIVED DENSITY'])


# In[84]:


df2.shape


# In[85]:


depth_train= np.linspace(2700,3495,len(df2))
df2['Depth']= depth_train


# In[86]:


df2


# In[87]:


sns.lineplot(data=df2['BEST CALIPER,AVERAGE DIAMETER'])


# In[88]:


df2.plot(subplots=True,figsize=(10,30))


# In[89]:


sns.lineplot(data=df2['IMAGE DERIVED PHOTOELECTRIC FACTOR'])


# In[90]:


sns.lineplot(data=df2['BEST THERMAL POROSITY'])


# In[91]:


sns.lineplot(data=df2['RESISTIVITY OF WATER FILLED FORMATION'])


# In[92]:


sns.lineplot(data=df2['RELATIVE PERMEABILITY TO HYDROCARBON'])


# In[93]:


sns.lineplot(data=df2['RELATIVE PERMEABILITY TO WATER'])


# In[94]:


sns.lineplot(data=df2['WATER CUT'])


# In[95]:


sns.lineplot(data=df2['MATRIX BULK DENSITY FROM ELEMENTAL CONCENTRATIONS'])


# In[96]:


sns.lineplot(data=df2['TOTAL POROSITY'])


# In[97]:


df3


# In[98]:


random_predict= model.predict(X_test)


# In[99]:


random_predict


# In[100]:


Y_test


# In[101]:


random_predict.shape


# In[102]:


preds= np.array(random_pred)


# In[103]:


reals= np.array(Y_test)


# In[104]:


r2= r2_score(Y_test,random_pred)
r2


# In[105]:


# Plot results:
plt.figure(figsize=(15,5))
i = 0
#plt.subplot(1,2,i+1)
plt.plot(preds, reals, '.', label = 'r^2 = %.3f' % ((r2_score(reals, preds))))
plt.plot([reals.min(),reals.max()],[reals.min(),reals.max()], 'r', label = '1:1 line')
plt.title('#1 Porosity: y_true vs. y_ estimate for Random Forest'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()


# In[106]:


preds= np.array(lr_pred)

# Plot results:
plt.figure(figsize=(15,5))
i = 0
#plt.subplot(1,2,i+1)
plt.plot(preds, reals, '.', label = 'r^2 = %.3f' % ((r2_score(reals, preds))))
plt.plot([reals.min(),reals.max()],[reals.min(),reals.max()], 'r', label = '1:1 line')
plt.title('#1 Porosity: y_true vs. y_ estimate for Linear Regression'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()


# In[107]:


preds= np.array(svm_pred)

# Plot results:
plt.figure(figsize=(15,5))
i = 0
#plt.subplot(1,2,i+1)
plt.plot(preds, reals, '.', label = 'r^2 = %.3f' % ((r2_score(reals, preds))))
plt.plot([reals.min(),reals.max()],[reals.min(),reals.max()], 'r', label = '1:1 line')
plt.title('#1 Porosity: y_true vs. y_ estimate for Linear Regression'); plt.xlabel('Estimate Porosity'); plt.ylabel('True Porosity')
plt.legend()


# In[108]:


from matplotlib import pyplot


# In[109]:


model.feature_importances_


# In[110]:


corr= df2.corr()


# In[111]:


sns.heatmap(corr,annot=True)


# In[112]:


importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[113]:


random_predict


# In[114]:


s1=pd.Series(model1.predict(X_test),name='predicted porosity',index=X_test.index)


# In[115]:


output1=pd.concat([X_test,Y_test],axis=1)


# In[116]:


output1


# In[117]:


sns.lineplot(data=Y_test)
sns.lineplot(data=s1,label='predicted porosity')


# In[118]:


model1.predict(X_train)


# In[119]:


s1=pd.Series(model1.predict(X_train),name='predicted porosity',index=X_train.index)


# In[120]:


output1=pd.concat([X_train,Y_train],axis=1)


# In[121]:


sns.lineplot(data=Y_train)
sns.lineplot(data=s1,label='predicted porosity')


# In[122]:


model.predict(X_test)


# In[123]:


s2=pd.Series(model.predict(X_test),name='predicted porosity',index=X_test.index)


# In[124]:


output2=pd.concat([X_test,Y_test],axis=1)


# In[125]:


sns.lineplot(data=Y_test)
sns.lineplot(data=s2,label='predicted porosity')


# In[126]:


model.predict(X_train)


# In[127]:


s2=pd.Series(model.predict(X_train),name='predicted porosity',index=X_train.index)


# In[128]:


output2=pd.concat([X_train,Y_train],axis=1)


# In[129]:


sns.lineplot(data=Y_train)
sns.lineplot(data=s2,label='predicted porosity')


# In[130]:


from sklearn.metrics import mean_squared_error


# In[131]:


mean_squared_error(X_test,Y_test)


# In[ ]:





# In[ ]:




