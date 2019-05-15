# Use this script to inspect the CSV file outputted from root2csv.py
import pandas as pd
import matplotlib.pyplot as plt
from features import column_names

df = pd.read_csv("csv/topreco.csv", names=column_names)
columns = df.columns
shape = df.shape
print("Shape of Data: {}".format(shape))
for i in range(len(columns)):
    plt.clf()
    fig = plt.figure()
    plt.hist(df[column_names[i]], bins=100, label=column_names[i])
    plt.legend()
    plt.title("Number of Events: {}".format(shape[0]))
    plt.ylabel("Counts")
    plt.savefig("Plots/{}.png".format(column_names[i]))
