try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import json
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please run: pip install pandas numpy matplotlib")
    sys.exit(1)

# for more info, read: https://developers.google.com/machine-learning/crash-course/linear-regression

class LinearRegression:
    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0
    
    def estimatePrice(self, mileage):
        print("theta0: ", self.theta0, "theta1: ", self.theta1)
        # regression line
        return self.theta0 + (self.theta1 * mileage)

    def performLinearRegression(self, m, x, y, x_min, x_max, y_min, y_max):  
        # min-max normalization
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)

        learning_rate = 0.01
        # at first we calculate just with y = 0*x + 0 (ax+b) to get the maximum loss
        it = 10000
        prev_loss = float('inf')
        # get a certain difference between the real value vs the computed value
        for i in range(it):
            # calculate the error at each step from the actual value
            estimate_price = self.theta0 + (self.theta1 * x_norm)
            error = estimate_price - y_norm

            # bias (theta0) and weight (theta1) derivative
            tmp_theta0 = learning_rate * (1 / m) * error.sum()
            tmp_theta1 = learning_rate * (1 / m) * (error * x_norm).sum()

            # get the new weight and bias 
            self.theta0 = self.theta0 - tmp_theta0
            self.theta1 = self.theta1 - tmp_theta1

            # calculate the MSE to get the loss and see if the algorithm shoudl stop
            loss = (error ** 2).mean()
            if abs(prev_loss - loss) < 1e-6:
                break
            prev_loss = loss
            
        mae = np.mean(np.abs(error))
        print(f"mae (how much the price predictions are off): {mae * (y_max - y_min)}")
        # denormalization of theta0 and theta1
        old_theta = self.theta1
        self.theta1 = self.theta1 * (y_max - y_min) / (x_max - x_min)
        self.theta0 = y_min + (y_max - y_min) * (self.theta0 - old_theta * x_min / (x_max - x_min))
        return
    
    def precision(self, prediction, real):
        pass

    def makeGraph(self, csv_path):
        df = pd.read_csv(csv_path)
        x = df["km"]
        y = df["price"]
        
        # my model 
        x = df["km"]
        y_pred = self.theta0 + (self.theta1 * x)
        df.plot(kind = 'scatter', x = 'km', y = 'price')
        plt.plot(x, y_pred, color='red', label='My Regression Line') 
    
        plt.legend()   
        plt.title("Price of a car for a given mileage")
        plt.xlabel("Kilometers")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return
    
    def save_model(self, filename="model.json"):
        data = {
            "theta0": self.theta0,
            "theta1": self.theta1
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="model.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                self.theta0 = data.get("theta0", 0.0)
                self.theta1 = data.get("theta1", 0.0)
        except FileNotFoundError:
            self.theta0 = 0.0
            self.theta1 = 0.0

class ReadFile:
    def __init__(self):
        pass

    def readCSV(self, csv_path):
        # csv : pandas.core.frame.DataFrame
        df = pd.read_csv(csv_path)
        # m : max rows (because that is the data we are going for)
        m = len(df)

        x = df["km"].copy()
        y = df["price"].copy()
        x_min, x_max = df["km"].min(), df["km"].max()
        y_min, y_max = df["price"].min(), df["price"].max()
        return m, x, y, x_min, x_max, y_min, y_max
    

def main(): 
    readFile = ReadFile()
    linearReg = LinearRegression()

    m, x, y, x_min, x_max, y_min, y_max = readFile.readCSV("./data.csv")
    linearReg.performLinearRegression(m, x, y, x_min, x_max, y_min, y_max)
    linearReg.save_model()
    linearReg.makeGraph("./data.csv")

if __name__ == "__main__":
    main()
