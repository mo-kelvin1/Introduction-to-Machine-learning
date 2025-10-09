import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


def predict_with_sklearn():
    df= pd.read_csv('testscores.csv')
    reg= LinearRegression()
    reg.fit(df[['math']],df['cs'])
    return reg.coef_, reg.intercept_

def gradient_descent(x,y):
    if(len(x)!= len(y)):
        print("the two arrays must be of the same length")
        return 
    m_curr = b_curr = 0
    learning_rate=0.0001
    iterations = 1000000
    n= len(x)
    cost_previous = 0
    for i in range(iterations):
        y_predicted= m_curr * x + b_curr
        cost_function= (1/n)*np.sum([value**2 for value in (y-y_predicted)])
        m_derivative = -(2/n)*np.sum(x*(y-y_predicted))
        b_derivative = -(2/n)*np.sum(y-y_predicted)
        m_curr = m_curr- learning_rate* m_derivative
        b_curr = b_curr -learning_rate*b_derivative
        if math.isclose(cost_function, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost_function
        print(f"m : {m_curr}, b: {b_curr}, iteration: {i}, cost : {cost_function} ")
    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv('testscores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_with_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))

