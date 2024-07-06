import pandas as pd
import urllib.request
import numpy as np
import matplotlib.pyplot as plt




class MooresLaw:
    def __init__(self) :
        self.data = 0
        self.url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"
        
    def load_data(self):

        with urllib.request.urlopen(self.url) as f:
            self.data = pd.read_csv(f)

        
        # Set column names for easier access
        self.data.columns = ["Year", "Transistor_Count"]
        
        # Convert the columns to numeric values
        self.data["Year"] = pd.to_numeric(self.data["Year"], errors='coerce')
        self.data["Transistor_Count"] = pd.to_numeric(self.data["Transistor_Count"], errors='coerce')

        

        x = self.data["Year"].values
        y = self.data["Transistor_Count"].values

        plt.scatter(x,y)
        plt.show()
        
        # Apply log transformation to y to linearize the exponential growth
        y = np.log(y)
        
        # Normalize x by subtracting the mean to center the data
        x = x - x.mean()

        return x,y

   
    def calculate(self):
        x,y = self.load_data()

        # apply the formula
        denominator = x.dot(x)-x.mean()* x.sum()
        a = (x.dot(y)-y.mean()*x.sum())/denominator
        b = (y.mean()*x.dot(x)-x.mean()*x.dot(y))/denominator

        print(a,b)

        predict = x.dot(a) + b
        # print(y,predict)

        print("time to double:",np.log(2)/a)

        numerator = y-predict
        den = y- y.mean()
        r_squarred = 1-(numerator.dot(numerator)/den.dot(den))
        print("r_squarred:",r_squarred)


        plt.scatter(x, y)
        plt.plot(x, predict)
        plt.show()


lR = MooresLaw()
lR.calculate()


        
        