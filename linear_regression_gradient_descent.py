# Author asethi
# Date 09/17/2019
#Importing packages
import researchpy as rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Implementing linear regression from scratch using batch gradient descent as the optimization algorithm.


def linear_regression_mains(X, y, alpha, num_iters,cost_threshold):
    theta = np.random.randn(X.shape[1])
    theta, cost, iter_runs = batch_gradient_descent_mains(theta,alpha,num_iters,X,y,cost_threshold)
    return theta, cost,iter_runs


# Function for batch gradient descent 
def batch_gradient_descent_mains(theta, alpha, num_iters,  X, Y,cost_threshold):
    #Initializing the array for cost calculations
    cost = np.ones(num_iters)
    
    for i in range(0,num_iters):
        #print(i)
          
        # Calculating hypothesis
        hypothesis_calculated = hypothesis_mains(theta, X)
        
        # Gradient is the differnce in y actual, calculated hypothesis, multiplied by 
        # learn rate and iteration count
        gradient = (alpha/X.shape[0]) * float(sum(hypothesis_calculated - Y))
        #updating theta 0 by subtracting gradient of this calculation
        
        theta[0] = theta[0] - gradient
        
        # for other thetas similar gradient calculation is done an the cost vals are simulataneously updated
        # by subtracting gradient
        for col in range(1,X.shape[1]):
            gradient_current = ((alpha/X.shape[0]) * float(sum(np.dot(X.T[col],(hypothesis_calculated-Y)))))
            theta[col] = theta[col] - gradient_current
        
        # cost of this iteration is calcuated simply by using the formula
        cost[i] = (1/X.shape[0]) * 0.5 * float(sum(np.square(hypothesis_calculated - Y)))
        
        # note : for i =0 we cannnot calculate the delta of costs, for the rest of i
        # we put a condition if the marginal change in cost is smaller than the given threshold exits out of 
        #iteration loop
        if(i>0):
            if(abs(cost[i]-cost[i-1])<cost_threshold):
                break
    #reshaping theta after final values are calculated
    theta = theta.reshape(1,X.shape[1])
    # total iterations ran are part ofthe output
    total_iterations_ran=i+1
    return theta, cost,total_iterations_ran


# Hypothesis calculation
# is similar to the dot product, it is estimated value of y bar
# for a given set of theta
def hypothesis_mains(theta, X):
    
    h_calc = np.ones((X.shape[0],1))
    
    theta = theta.reshape(1,X.shape[1])
    
    for i in range(0,X.shape[0]):
        h_calc[i] = float(np.dot(theta,X.T[:,i]))
        
    h_calc = h_calc.reshape(X.shape[0],1)
    return h_calc


# function to prepare data for linear regression evoked
X_train_mains,Y_train_mains,X_test_mains,Y_test_mains = data_prep_linear_regression(energy_source_data_features[XMaster_Updated],energy_source_data_features[YMaster])
# iter_max=1000
# cost_threshold=0.02
# optimal_weight, cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.15,iter_max, cost_threshold)


#prediction of test set and test set rmse calculation on the test set
def regression_prediction(optimal_weights,dep_vars,indep_vars):
    # we take optimal weight and dep vars as inputs and compute the dot product to calculate the predicted values
    predicted_values=dep_vars.dot(optimal_weights.transpose())
    #pred_values=predicted_values.to_numpy()
    #original_test_values=test_set['Appliances'].to_numpy()
    #Initializing RMSE
    rmse=0
    m=dep_vars.shape[0]
    
    # Differencing, squaring the differnec in prediction and actual and summing it all together,
    # before mean normalizing and calculating the root
    
    # rmse and predicted vars returned
    for i in range(m):
        rmse += (indep_vars[i] - predicted_values[i]) ** 2
    
    rmse = np.sqrt(rmse/m)
    return predicted_values,rmse

# predicted_values_train,rmse_train=regression_prediction(optimal_weight,X_train_mains,Y_train_mains)
# predicted_values_test,rmse_test=regression_prediction(optimal_weight,X_test_mains,Y_test_mains)

# print(rmse_train)
# print(rmse_test)
###
###
## Experiment  linear regression

# Max iterations are 1000, no threshold set for exp1
iter_max=1000
cost_threshold=0.00

# trace of cost array for all iterations intialized
cost_trace = np.array([])

# calculating cost trace for all learn rates
for learn_rate in [0.01,0.1,0.2,0.25]:
    
    optimal_weight,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, cost_threshold)
    cost_trace = np.append(cost_trace, cost)

###
### Invoking the same functions for experiment 1 for test data
iter_max=1000
cost_threshold=0.00
cost_trace = np.array([])
for learn_rate in [0.01,0.1,0.2,0.25]:
    
    optimal_weight,cost,iter_runs=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, cost_threshold)
    cost_trace = np.append(cost_trace, cost)
    

# Running Gradient Descent and calculating train and test set error for learn rate 0.01 
optimal_weight_alpha1,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.01,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha1 = regression_prediction(optimal_weight_alpha1,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha1 = regression_prediction(optimal_weight_alpha1,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.01 == ",rmse_train_alpha1)
print ("RMSE test for learn rate 0.01 == ",rmse_test_alpha1)


# Running Gradient Descent and calculating train and test set error for learn rate 0.1
optimal_weight_alpha2,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.1,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha2 = regression_prediction(optimal_weight_alpha2,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha2 = regression_prediction(optimal_weight_alpha2,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.1 == ",rmse_train_alpha2)
print ("RMSE test for learn rate 0.1 == ",rmse_test_alpha2)



# Running Gradient Descent and calculating train and test set error for learn rate 0.2
optimal_weight_alpha3,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.2,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha3 = regression_prediction(optimal_weight_alpha3,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha3 = regression_prediction(optimal_weight_alpha3,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.2 == ",rmse_train_alpha3)
print ("RMSE test for learn rate 0.2 == ",rmse_test_alpha3)



# Running Gradient Descent and calculating train and test set error for learn rate 0.25
optimal_weight_alpha4,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.25,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha4 = regression_prediction(optimal_weight_alpha4,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha4 = regression_prediction(optimal_weight_alpha4,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.25 == ",rmse_train_alpha4)
print ("RMSE test for learn rate 0.25 == ",rmse_test_alpha4)

