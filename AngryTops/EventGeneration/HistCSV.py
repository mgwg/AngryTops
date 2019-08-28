# Use this script to inspect the CSV file outputted from root2csv.py
import pandas as pd
import matplotlib.pyplot as plt
from AngryTops.features import column_names

def getRawHists(fname):
    df = pd.read_csv(fname, names=column_names)
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
        plt.yscale('log')
        plt.savefig("Plots/topreco_5dec3/{0}.png".format(column_names[i]))
        

if __name__ == "__main__":
    getRawHists(fname='csv/topreco_5dec3.csv')
