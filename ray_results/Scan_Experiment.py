import json
import os
print(os.getcwd())

# Change this depending on the experiment name
filename = 'experiment_state-2019-07-04_18-32-14.json'

with open(filename, 'rb') as f:
    datastore = json.load(f)

# Find and store the best model
best_mse = 10e5
best_model = None

# Go through all of the models in the experiment
for i in range(len(datastore['checkpoints'])):
    if datastore['checkpoints'][i]['last_result']['mse'] < best_mse:
        best_mse = datastore['checkpoints'][i]['last_result']['mse']
        best_model = datastore['checkpoints'][i]['config']

# Print out best result
print("Best MSE: %.5f" % best_mse)
print("Best Model: ", best_model)
