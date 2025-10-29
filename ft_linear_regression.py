import pandas as pd
import numpy as np
from scipy import stats
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

# for more info, read: https://developers.google.com/machine-learning/crash-course/linear-regression

def estimatePrice(mileage, theta0, theta1):
    print("theta0: ", theta0, "theta1: ", theta1)
    # regression line
    return theta0 + (theta1 * mileage)

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)

def readCSV(csv_path):
    # csv : pandas.core.frame.DataFrame
    df = pd.read_csv(csv_path)
    # m : max rows (because that is the data we are going for)
    m = len(df)

    x = df["km"].copy()
    y = df["price"].copy()
    x_min, x_max = df["km"].min(), df["km"].max()
    y_min, y_max = df["price"].min(), df["price"].max()
    return m, x, y, x_min, x_max, y_min, y_max

def performLinearRegression(m, x, y, x_min, x_max, y_min, y_max):  
    # min-max normalization
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    learning_rate = 0.1
    # at first we calculate just with y = 0*x + 0 (ax+b) to get the maximum loss
    theta0 = theta1 = 0
    it = 10000
    prev_loss = float('inf')
    # get a certain difference between the real value vs the computed value
    for i in range(it):
        # calculate the error at each step from the actual value
        estimate_price = theta0 + (theta1 * x_norm)
        error = estimate_price - y_norm

        # bias (theta0) and weight (theta1) derivative
        tmp_theta0 = learning_rate * (1 / m) * error.sum()
        tmp_theta1 = learning_rate * (1 / m) * (error * x_norm).sum()

        # get the new weight and bias 
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        # calculate the MSE to get the loss and see if the algorithm shoudl stop
        loss = (error ** 2).mean()
        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss
        
    # denormalization of theta0 and theta1

    b1 = theta1 * (y_max - y_min) / (x_max - x_min)
    b0 = y_min + (y_max - y_min) * (theta0 - theta1 * x_min / (x_max - x_min))

    return b0, b1

def makeGraph(csv_path, theta0, theta1):
    df = pd.read_csv(csv_path)
    x = df["km"]
    y = df["price"]
    
    # Geeks for Geeks model
    slope, intercept, r, p, std_err = stats.mstats.linregress(x, y)
    def myfunc(x):
        return slope * x + intercept
    mymodel = list(map(myfunc, x))
    df.plot(kind = 'scatter', x = 'km', y = 'price')
    b = estimate_coef(df["km"], df["price"])
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color='purple', label='Regression Line Geeks')
    
    # my model  
    x = df["km"]
    y_pred = theta0 + (theta1 * x)
    plt.plot(x, y_pred, color='red', label='My Regression Line') 
 
    plt.legend()   
    plt.title("Price of a car for a given mileage")
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

mileage = int(input("Please input the mileage: "))
m, x, y, x_min, x_max, y_min, y_max = readCSV("./data.csv")
theta0, theta1 = performLinearRegression(m, x, y, x_min, x_max, y_min, y_max)
estimate_price = estimatePrice(mileage, theta0, theta1)
makeGraph("./data.csv", theta0, theta1)
print("estimated price", estimate_price)



