# Use this script to inspect the CSV file outputted from root2csv.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv/topreco.csv")
columns = df.columns
shape = df.shape
print("Shape of Data: {}".format(shape))
for i in range(len(columns)):
    plt.hist(data[columns[i]], bins=100, label=columns[i])
    plt.legend()
    plt.title("Number of Events: {}".format(shape[0]))
    plt.ylabel("Counts")
    plt.savefig("Plots/{}".format(columns[i]))
