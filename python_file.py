# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# %%
csv_data = pd.read_csv('cardata.csv')
intial_df = pd.DataFrame(csv_data)
print(intial_df)

# %%
intial_df.describe()

# %%
working_df = intial_df.copy()
working_df.drop(columns='Car_Name',inplace=True)
print(working_df)

# %%
max_year= np.max(working_df['Year'])
working_df.insert(1,column='Age',value = 0)
working_df['Age'] = max_year - working_df['Year']
print(working_df)

# %%
working_df.describe()

# %%
X = pd.DataFrame(working_df,columns=['Age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
Y = working_df['Selling_Price'].values.reshape(-1,1)
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Fuel_Type', 'Seller_Type', 'Transmission']]), columns=encoder.get_feature_names_out(['Fuel_Type', 'Seller_Type', 'Transmission']))
X.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], axis=1, inplace=True)
X = pd.concat([X, X_encoded], axis=1)
print(X)



# %%
correlation_df = pd.DataFrame()
columns = list(X.columns)
for i in range(0,3) :
    for j in range(i+1,4) :
        correlation_factor = np.corrcoef(X.iloc[:,i],X.iloc[:,j])
        correlation_df.loc[columns[i],columns[j]] = correlation_factor[0,1]

print(correlation_df)        


# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
regressor = LinearRegression()

# %%
KFold_validation = KFold(4)
result_1 = cross_val_score(regressor,X,Y,cv=KFold_validation)
print(result_1)
print(np.mean(result_1))

# %%
regressor.fit(X_train,Y_train)
result_2 = regressor.score(X_test,Y_test)
print(result_2)
Y_predict = regressor.predict(X_test)

# %%
plt.scatter(Y_test,Y_predict)
plt.title('Y_test Vs Y_prediction')
plt.xlabel('Y_test')
plt.ylabel('Y_predict')
plt.grid()
plt.show()

# %%
X_test
X_test.insert(8,'Y_test',Y_test)
X_test.insert(9,'Y_predict',Y_predict)
age_sorted_df = X_test.sort_values(by='Age')
print(age_sorted_df)

# %%
intercept = regressor.intercept_
coefficient = regressor.coef_
print(f'Intercept is equal to : {(intercept)} and Coefficient is equal to : {(coefficient)}')

# %%
compare_df = pd.DataFrame({'Actual':Y_test.flatten(),'Prediction':Y_predict.flatten()})
print(compare_df)

# %%
print('Mean absolut Errro: ' , metrics.mean_absolute_error(Y_test,Y_predict))
print('Mean squared Error: ', metrics.mean_squared_error(Y_test,Y_predict))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(Y_test,Y_predict)))
print('R2 Score: ',metrics.r2_score(Y_test,Y_predict))


