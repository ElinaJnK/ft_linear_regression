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

def performLinearRegression(csv_path):
    # csv : pandas.core.frame.DataFrame
    df = pd.read_csv(csv_path)
    # m : max rows (because that is the data we are going for)
    m = len(df)

    km = df["km"]
    price = df["price"]
    # min-max normalization
    df["km"] = (df["km"] - df["km"].min()) / (df["km"].max() - df["km"].min())
    #df["price"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())

    learning_rate = 0.0000001
    # at first we calculate just with y = 0*x + 0 (ax+b) to get the maximum loss
    theta0 = theta1 = 0
    it = 1000
    prev_loss = float('inf')
    # get a certain difference between the real value vs the computed value
    for _ in range(it):
        # calculate the error at each step from the actual value
        estimate_price = theta0 + (theta1 * df["km"])
        error = estimate_price - df["price"]

        # bias (theta0) and weight (theta1) derivative
        tmp_theta0 = learning_rate * (1 / m) * error.sum()
        tmp_theta1 = learning_rate * (1 / m) * (error * df["km"]).sum()

        # get the new weight and bias 
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        #theta0 = abs(tmp_theta0)
        #theta1 = abs(tmp_theta1)

        # calculate the MSE to get the loss and see if the algorithm shoudl stop
        loss = (error ** 2).mean()
        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss
    return theta0, theta1

def makeGraph(csv_path, theta0, theta1):
    # plotting
    df = pd.read_csv(csv_path)
    x = df["km"]
    print("x:", x)
    print("theta0: ", theta0, "theta1: ", theta1)
    y = theta0 + (theta1 * x)
    print("y:", y)
    #what I should have
    slope, intercept, r, p, std_err = stats.mstats.linregress(x, df["price"])
    def myfunc(x):
        return slope * x + intercept
    mymodel = list(map(myfunc, x))
    # what is in the file
    df.plot(kind = 'scatter', x = 'km', y = 'price')
    
    # test
    b = estimate_coef(df["km"], df["price"])
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color='purple', label='Regression Line Geeks')
    
    plt.plot(x, y, color='red', label='Regression Line')
    #plt.plot(x, mymodel, color='blue', label='Regression Line Real')
    plt.title("Price of a car for a given mileage")
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

mileage = int(input("Please input the mileage: "))
theta0, theta1 = performLinearRegression("./data.csv")
estimate_price = estimatePrice(mileage, theta0, theta1)
makeGraph("./data.csv", theta0, theta1)
print("estimated price", estimate_price)



