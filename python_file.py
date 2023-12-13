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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# %%
csv_data = pd.read_csv('cardata.csv')
intial_df = pd.DataFrame(csv_data)
intial_df

# %%
intial_df.describe()

# %%
working_df = intial_df.copy()
working_df.drop(columns='Car_Name',inplace=True)
working_df

# %%
max_year= np.max(working_df['Year'])
working_df.insert(1,column='Age',value = 0)
working_df['Age'] = max_year - working_df['Year']
working_df

# %%
working_df.describe()

# %%
X = pd.DataFrame(working_df,columns=['Age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
Y = working_df['Selling_Price'].values.reshape(-1,1)
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Fuel_Type', 'Seller_Type', 'Transmission']]), columns=encoder.get_feature_names_out(['Fuel_Type', 'Seller_Type', 'Transmission']))
X.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], axis=1, inplace=True)
X = pd.concat([X, X_encoded], axis=1)
X

# %%
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X_scaled

# %%
correlation_df = pd.DataFrame()
columns = list(X_scaled.columns)
for i in range(0,7) :
    for j in range(i+1,8) :
        correlation_factor = np.corrcoef(X_scaled.iloc[:,i],X_scaled.iloc[:,j])
        correlation_df.loc[columns[i],columns[j]] = correlation_factor[0,1]

correlation_df  


# %%
regressor = LinearRegression()

# %%
KFold_validation = KFold(n_splits=9,shuffle=True,random_state=42)
result_1 = cross_val_score(regressor,X_scaled,Y,cv=KFold_validation)
print(result_1)
print(np.mean(result_1))

# %%
X_train_1,X_test_1,Y_train_1,Y_test_1 = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

# %%
regressor.fit(X_train_1,Y_train_1)
result_2 = regressor.score(X_test_1,Y_test_1)
print(result_2)
Y_predict = regressor.predict(X_test_1)

# %%
plt.scatter(Y_test_1,Y_predict)
plt.title('Y_test Vs Y_prediction')
plt.xlabel('Y_test')
plt.ylabel('Y_predict')
plt.grid()
plt.show()

# %%
intercept = regressor.intercept_
coefficient = regressor.coef_
print(f'Intercept is equal to : {(intercept)} and Coefficient is equal to : {(coefficient)}')
X_scaled

# %%
compare_df = pd.DataFrame({'Actual':Y_test_1.flatten(),'Prediction':Y_predict.flatten()})
compare_df

# %%
print('Mean absolut Errro: ' , metrics.mean_absolute_error(Y_test_1,Y_predict))
print('Mean squared Error: ', metrics.mean_squared_error(Y_test_1,Y_predict))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(Y_test_1,Y_predict)))
print('R2 Score: ',metrics.r2_score(Y_test_1,Y_predict))

# %%
X_copy = X_scaled.copy()
Poly_Present_Price = X_scaled.Present_Price**3
X_copy.insert(2,'Poly_Present_Price',value=Poly_Present_Price)
X_train_2,X_test_2,Y_train_2,Y_test_2 = train_test_split(X_copy,Y,test_size=0.2,random_state=42)
regressor.fit(X_train_2,Y_train_2)
Y_predict = regressor.predict(X_test_2)


# %%
plt.scatter(Y_test_2,Y_predict)
plt.title('Y_test Vs Y_prediction')
plt.xlabel('Y_test')
plt.ylabel('Y_predict')
plt.grid()
plt.show()

# %%
intercept = regressor.intercept_
coefficient = regressor.coef_
print(f'Intercept is equal to : {(intercept)} and Coefficient is equal to : {(coefficient)}')
X_copy

# %%
compare_df = pd.DataFrame({'Actual':Y_test_2.flatten(),'Prediction':Y_predict.flatten()})
compare_df

# %%
print('Mean absolut Errro: ' , metrics.mean_absolute_error(Y_test_2,Y_predict))
print('Mean squared Error: ', metrics.mean_squared_error(Y_test_2,Y_predict))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(Y_test_2,Y_predict)))
print('R2 Score: ',metrics.r2_score(Y_test_2,Y_predict))

# %%
poly = PolynomialFeatures(degree=2)
polynomial_features = poly.fit_transform(X_scaled)
X_train_3,X_test_3,Y_train_3,Y_test_3 = train_test_split(polynomial_features,Y,test_size=0.2,random_state=42)
regressor.fit(X_train_3,Y_train_3)
Y_predict = regressor.predict(X_test_3)


# %%
plt.scatter(Y_test_3,Y_predict)
plt.title('Y_test Vs Y_prediction')
plt.xlabel('Y_test')
plt.ylabel('Y_predict')
plt.grid()
plt.show()


# %%
intercept = regressor.intercept_
coefficient = regressor.coef_
print(f'Intercept is equal to : {(intercept)} and Coefficient is equal to : {(coefficient)}')


# %%
compare_df = pd.DataFrame({'Actual':Y_test_3.flatten(),'Prediction':Y_predict.flatten()})
compare_df


# %%
print('Mean absolut Errro: ' , metrics.mean_absolute_error(Y_test_3,Y_predict))
print('Mean squared Error: ', metrics.mean_squared_error(Y_test_3,Y_predict))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(Y_test_3,Y_predict)))
print('R2 Score: ',metrics.r2_score(Y_test_3,Y_predict))

# %%
regressor.fit(polynomial_features,Y)

# %%



