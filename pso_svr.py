import pandas as pd
import numpy as np
import random
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from pyswarm import pso

seed = 42 #Fix seed for repeatability purposes
np.random.seed(seed)
random.seed(seed)
#==================================================================================
#Define performance indicators
#RMSE
def rmseCalc(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred) #Actual, predicted Trg values
    rmse = np.sqrt(mse)
    return rmse

#SMAPE
def smapeCalc(yTrue, yPred):
    y_true = np.array(yTrue)
    y_pred = np.array(yPred)
    
    epsilon = 1e-8
    denominator = ((np.abs(y_true)+np.abs(y_pred))/2.0)+epsilon
    numerator = np.abs(y_true-y_pred)
    smape = np.mean(numerator/denominator) * 100
    return smape

#MAPE
def mapeCalc(yTrue, yPred):
    y_true = np.array(yTrue)
    y_pred = np.array(yPred)

    # Avoid division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return float('inf')  # or raise an error if preferred

    mape_values = np.abs((y_true[non_zero_mask]-y_pred[non_zero_mask])/y_true[non_zero_mask])
    mape = np.mean(mape_values)*100
    return mape
#==================================================================================
#Get dataset
filePath = 'Metallic_Glass_Forming_with_features2.xlsx'
df = pd.read_excel(filePath) #Create datafile
#----------------------------------------------------------------------------------
#Separate dataframe into (groups of) parameters
constElement = df.iloc[:,2:48]
totalPerc = df['Total']
trg = df['Trg']
#----------------------------------------------------------------------------------
#Adjust percentage values of constituent elements so that sum is 100%
stdElement = constElement.div(totalPerc, axis=0).fillna(0)
#----------------------------------------------------------------------------------
#Remove Trg data with NaN
nanRows = trg[trg.isnull()].index.tolist() #Get indices of rows in Trg with NaN

stdElementNew = stdElement.drop(nanRows,axis=0)
trgNew = trg.drop(nanRows,axis=0)
#----------------------------------------------------------------------------------
#Create training and testing datasets
XTrain, XTest, yTrain, yTest = train_test_split(stdElementNew, trgNew,test_size=0.2, random_state=42)
#----------------------------------------------------------------------------------
#Standardize each dataframe column
scaler = StandardScaler()

#Fit and transform training dataset inputs
scaledTrain = scaler.fit_transform(XTrain)
XTrainScaled = pd.DataFrame(scaledTrain, columns=stdElement.columns)

#Transform only for testing dataset inputs
scaledTest = scaler.transform(XTest)
XTestScaled = pd.DataFrame(scaledTest, columns=stdElement.columns)
#==================================================================================
#Define objective function
def svr_cv(params):
    CValue, epsilonValue, gammaValue = params
    model = SVR(kernel='rbf', C=CValue, epsilon=epsilonValue, gamma=gammaValue)
    score = cross_val_score(
        model,
        XTrainScaled,
        yTrain,
        cv=5,
        scoring='neg_mean_squared_error')
    return -np.mean(score)

#Typical C range is 0.1 to 1000; typical epsilon range is 0.01 to 1
#Default gamma in Python is gamma = 1/(N x Var(X)), N = No. of features, X = input

lowerBound = [0.1, 0.01, 0.0001] #C, epsilon, gamma
upperBound = [100, 1, 1] #C, epsilon, gamma
#----------------------------------------------------------------------------------
#SVR model training and testing via PSO
bestParam, bestScore = pso(
    svr_cv,
    lowerBound,
    upperBound,
    swarmsize=30,
    maxiter=50,
    omega=0.5,
    phip=0.5,
    phig=0.5,
    debug=True)

print("Best C:", bestParam[0])
print("Best Epsilon:", bestParam[1])
print("Best Gamma:", bestParam[2])
print("Best Score:", bestScore)
#==================================================================================
#Calculate number of parameters and memory
modelFinal = SVR(kernel='rbf', C=bestParam[0], epsilon=bestParam[1], gamma=bestParam[2])
modelFinal.fit(XTrainScaled,yTrain)

num_support_vectors = len(modelFinal.support_)
num_parameters = num_support_vectors+1  # +1 for intercept
support_vectors_mem = modelFinal.support_vectors_.nbytes
dual_coef_mem = modelFinal.dual_coef_.nbytes
intercept_mem = sys.getsizeof(modelFinal.intercept_)

total_mem_bytes = support_vectors_mem+dual_coef_mem+intercept_mem
total_mem_kb = total_mem_bytes/1024
#----------------------------------------------------------------------------------
#Calculate RMSE, SMAPE, and MAPE
predictedTrg = modelFinal.predict(XTestScaled)

rmse = rmseCalc(yTest, predictedTrg) #Actual, predicted Trg values
smape = smapeCalc(yTest, predictedTrg) #Actual, predicted Trg values
mape = mapeCalc(yTest, predictedTrg) #Actual, predicted Trg values

print("Total Parameters:", num_parameters)
print("Total Memory:", total_mem_kb, "KB")
print('RMSE: ', rmse)
print('SMAPE: ', smape)
print('MAPE: ', mape)

#==================================================================================
