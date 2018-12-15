import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if  __name__ == "__main__":
    path='/Users/chensijin/Documents/git/ML/LinearRegression/Advertising.csv'

    
    data = pd.read_csv(path)
    x = data[['TV', 'Radio']]
    #x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(x)
    print(x.shape)
    print("##############################")
    print(y)
    
    #p = np.loadtxt(path,delimiter=',',skiprows=1)
    #print(p)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    print("===================")
    print(x_train)
    print(y_train)
    
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_) #xi shu
    print(linreg.intercept_) #jie ju
    
    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2) # jun fang wu cha, mean squred error
    rmse = np.sqrt(mse) #jun fang gen wu cha, root mean squred error
    print(mse , rmse)