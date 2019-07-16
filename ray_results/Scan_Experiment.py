import json
import os
import sys
print(os.getcwd())

# Change this depending on the experiment name
filename = sys.argv[1]

with open(filename, 'rb') as f:
    datastore = json.load(f)

# Find and store the best model
best_mse = 10e5
best_model = None

# Go through all of the models in the experiment
for i in range(len(datastore['checkpoints'])):
    try:
        if datastore['checkpoints'][i]['last_result']['mse'] < best_mse:
            best_mse = datastore['checkpoints'][i]['last_result']['mse']
            best_model = datastore['checkpoints'][i]['config']
    except Exception as e:
        print(e)
        print("Skipping trial")

# Print out best result
print("Best MSE: %.5f" % best_mse)
print("Best Model: ", best_model)
