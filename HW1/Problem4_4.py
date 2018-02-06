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
    data_train = data[:400]
    data_test = data[400:]
    target_train = target[:400]
    target_test = target[400:] 
    x_train = x[:400]
    x_test = x[400:]
    reg.fit(data_train,target_train)
    print("Here are coefficients: ", reg.coef_ )
    predictions_train = reg.predict(data_train)
    predictions_test = reg.predict(data_test)
    #print("target: ", target)
    #print("predictions: ", predictions)
    print("Mean squared error train: " , mean_squared_error(target_train, predictions_train))
    print("Mean squared error test: " , mean_squared_error(target_test, predictions_test))
    residuals_train = np.subtract(target_train,predictions_train)
    residuals_train = np.absolute(residuals_train)
    plt.plot(x_train,residuals_train)
    plt.ylabel("Residual Train")
    plt.xlabel("Sample Number")
    plt.show()
    residuals_test = np.subtract(target_test,predictions_test)
    residuals_test = np.absolute(residuals_test)
    plt.plot(x_test,residuals_test)
    plt.ylabel("Residual Test")
    plt.xlabel("Sample Number")
    plt.show()
    #print("x: ", x)
    #print("residuals: ", residuals)
          	 
            

