#wine quality dataset for multiple linear regression using sklearn
#from https://www.geeksforgeeks.org/dataset-for-linear-regression/

#multi-colinearity analysis using VIF https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

#data ini /Users/ivan.hom/projects/red_wine_quality/winequality_red.csv

#STEP1) run at command line to setup
#pip3 install xgboost
#pip3 install scikit-learn
#pip3 install matplotlib
#pip3 install seaborn
#pip3 install pandas
#pip3 install numpy
#pip3 install statsmodels 
#brew install libomp

#STEP2) load packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model



#STEP3 read into pandas dataframe
dataset = pd.read_csv('/Users/ivan.hom/projects/red_wine_quality/winequality_red.csv')

dataset

dataset.dtypes

#show if null
dataset.info()

#show IQR stats
dataset.describe()


#STEP4) build multiple regression model and calcuate residuals
#target variable is quality

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = dataset['quality']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#make predictions
y_hat = regr.predict(X)

#convert pandas array to dataframe
y_hat_df = pd.DataFrame(y_hat)

#calc residuals
residuals = y.subtract(y_hat_df, fill_value=0)
final_residuals = residuals.iloc[:, 0] 


# Summarize the fit of the model
mse = np.mean((final_residuals)**2)
print(regr.intercept_, regr.coef_, mse) 

#21.965208449448745 [ 2.49905527e-02 -1.08359026e+00 -1.82563948e-01  1.63312698e-02 -1.87422516e+00  4.36133331e-03 -3.26457970e-03 -1.78811638e+01 -4.13653144e-01  9.16334413e-01  2.76197699e-01] 0.6395180110150008


#STEP5) check if residuals are distributed normally and no bias

plt.hist(final_residuals)
plt.show()

#yes is normal and there is no bias


#STEP6) determine multi-colinearity of input features by checking variance inflation factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
vif_df = calculate_vif(dataset, features)
print(vif_df)

#                 feature          VIF
#0          fixed acidity    74.452265
#1       volatile acidity    17.060026
#2            citric acid     9.183495
#3         residual sugar     4.662992
#4              chlorides     6.554877
#5    free sulfur dioxide     6.442682
#6   total sulfur dioxide     6.519699
#7                density  1479.287209
#8                     pH  1070.967685
#9              sulphates    21.590621
#10               alcohol   124.394866


#Interpreting VIF values:
#VIF = 1: No multicollinearity
#1 < VIF < 5: Moderate multicollinearity
#VIF >= 5: High multicollinearity

#drop features with over VIF over 10
X.drop(['fixed acidity', 'volatile acidity', 'density', 'pH', 'sulphates', 'alcohol'], axis=1, inplace=True)

regr = linear_model.LinearRegression()
regr.fit(X, y)

#make predictions
y_hat = regr.predict(X)

#convert pandas array to dataframe
y_hat_df = pd.DataFrame(y_hat)

#calc residuals
residuals = y.subtract(y_hat_df, fill_value=0)
final_residuals = residuals.iloc[:, 0]


# Summarize the fit of the model
mse = np.mean((final_residuals)**2)
print(regr.intercept_, regr.coef_, mse)

#5.701840486253058 [ 1.16318808  0.00748543 -2.97757875  0.01274554 -0.00735808] 0.4913269641325113

#better, mse decreased so model is more accurate





