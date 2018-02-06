import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


filename = "dow_jones_index.data"
i = 0
data = []
target = []
x = []
num = 0
with open(filename, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if i != 0:
            if len(row[3]) == 0: continue
            if len(row[4]) == 0: continue
            if len(row[5]) == 0: continue
            if len(row[6]) == 0: continue
            if len(row[7]) == 0: continue
            if len(row[8]) == 0: continue
            if len(row[9]) == 0: continue
            row[3] = row[3].replace("$","")
            row[4] = row[4].replace("$","")
            row[5] = row[5].replace("$","")
            row[6] = row[6].replace("$","")
            row[11] = row[11].replace("$","")
            if len(row[11]) == 0: continue
            sample = [float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]),
                 float(row[8]), float(row[9])] 
            data.append(sample) 
            target.append(float(row[11])) #next_weeks_open     
            x.append(num)
            num = num + 1
        i = i + 1
    reg = linear_model.LinearRegression()
    reg.fit(data,target)
    print("Here are coefficients: ", reg.coef_ )
    predictions = reg.predict(data)
    #print("target: ", target)
    #print("predictions: ", predictions)
    print("Mean squared error: " , mean_squared_error(target, predictions))
    residuals = np.subtract(target,predictions)
    residuals = np.absolute(residuals)
    plt.plot(x,residuals)
    plt.ylabel("Residual")
    plt.xlabel("Sample Number")
    plt.show()
    #print("x: ", x)
    #print("residuals: ", residuals)
          	 
            

