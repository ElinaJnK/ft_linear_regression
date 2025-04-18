import pandas as pd
import numpy as np
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

# for more info, read: https://developers.google.com/machine-learning/crash-course/linear-regression

def estimatePrice(mileage, theta0, theta1):
    print("theta0: ", theta0, "theta1: ", theta1)
    return theta0 + (theta1 * mileage)


def performLinearRegression(csv_path):
    # csv : pandas.core.frame.DataFrame
    df = pd.read_csv(csv_path)
    # m : max rows
    m = len(df)

    km = df["km"]
    price = df["price"]
    # min-max normalization
    df["km"] = (df["km"] - df["km"].min()) / (df["km"].max() - df["km"].min())
    #df["price"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())

    learning_rate = 0.0000001
    # at first we calculate just with y = 0*x + 0 (ax+b) to get the maximum loss
    theta0 = theta1 = 0
    it = 100
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
    y = theta0 + (theta1 * x)
    df.plot(kind = 'scatter', x = 'km', y = 'price')
    plt.plot(x, y, color='red', label='Regression Line')
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



