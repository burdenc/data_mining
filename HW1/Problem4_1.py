import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

print("Hello")

filename = "dow_jones_index.data"
#file = open(filename, "r")
i = 0
data = []
target = []
with open(filename, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if i != 0:
            print("row[8]: ", row[8], " row[9]: ", row[9], " row[11]: ", row[11])
            if len(row[8]) == 0: continue
            if len(row[9]) == 0: continue
            row[11] = row[11].replace("$","")
            if len(row[11]) == 0: continue
            sample = [float(row[8]), float(row[9])] #row[8]: percent_change_price, row[9]: percent_change_volume_over_last_wk
            data.append(sample) #percent_change_price
            target.append(float(row[11])) #next_weeks_open     
        i = i + 1
    print("data[0]: ", data, " target[0]", target)
    reg = linear_model.LinearRegression()
    reg.fit(data,target)
    print("1 Here are coef: ", reg.coef_ )
    predictions = reg.predict(data)
    print("Mean squared error: %.2f" , mean_squared_error(target, predictions))
    #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    #print("2 Here are coef: ", reg.coef_ )
    #reg.fit(data,target)
    #print("3 Here are coef: ", reg.coef_ )
      	 
            

