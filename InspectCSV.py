# Use this script to inspect the CSV file outputted from root2csv.py
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("csv/topreco.csv")
print(data.columns)
