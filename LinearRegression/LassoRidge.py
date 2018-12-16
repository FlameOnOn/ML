import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    path='/Users/chensijin/Documents/git/ML/Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    #print(x)
    #print(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    model = Lasso()
    alpha_can = np.logspace(-3,2,10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x,y)
    print("best params: ", lasso_model.best_params_)
    
    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2) 
    rmse = np.sqrt(mse)
    
    print(mse , rmse)
    
    modelR = Ridge()
    ridge_model = GridSearchCV(modelR, param_grid={'alpha': alpha_can}, cv=5)
    ridge_model.fit(x,y)
    
    print("best params for ridge model: ", ridge_model.best_params_)
    
    y_predict = ridge_model.predict(np.array(x_test))
    mse_ridge = np.average((y_predict-np.array(y_test)) ** 2)
    rmse_ridge = np.sqrt(mse_ridge)
    
    print("mse and rmse for ridge model is: " , mse_ridge , rmse_ridge)
    
    
#https://blog.csdn.net/jiang_jinyue/article/details/78369088
    