# Main.py

import pandas as pd
import numpy as np
import os
from digital_twin.data_processing import load_and_process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from biogears_python.execution import run_biogears
from biogears_python.xmlscenario import segments_to_xml
from sklearn.metrics import mean_absolute_error

# import lightgbm as lgb
# import ydf

# Define directories
data_dir = '/opt/biogears/core/build/runtime/simulation_results/'
outputs_dir = 'outputs'
model_output_dir = outputs_dir+'/models'
scalers_output_dir = outputs_dir+'/scalers'
visualizations_dir = outputs_dir+'/visualizations'
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(scalers_output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
output_csv_path = os.path.join(outputs_dir, 'processed_data.csv')

# Define feature and target columns
feature_cols = [
    'time_delta',
    'SkinTemperature(degC)',
    'intensity', 
    'atemp_c', 
    'rh_pct'
]
target_cols = [
    'SkinTemperature(degC)_diff'
]

# Define initial state of model
initial_state = (
    1,
    33,
    0.25,
    22.0,
    90.0
)

# Define the exercise trial for the testing of the model run function
n_segments = 10
segments_cool_humid={
    'time': [1.00] * n_segments,
    'intensity': [0.25] * n_segments,
    'atemp_c': [30.00] * n_segments,
    'rh_pct': [30.00] * n_segments
}

# Process the data and save it to a CSV file
results = load_and_process_data(
    data_dir=data_dir, 
    output_csv_path=output_csv_path,
    feature_cols=feature_cols,
    target_cols=target_cols,
    diff=True,
    time_deltas=True
)

df = results['df']

model = RandomForestRegressor()
model.fit(df[feature_cols], df[target_cols[0]])
pred = model.predict(np.array([initial_state]))

preds = []

state = initial_state
for i in range(len(segments_cool_humid['time'])):
    # Construct new state by adding the predicted deltas
    state_td  = state[0] + 1
    print(state[1], pred[0])
    state_st = state[1] + pred[0]
    state_int = segments_cool_humid['intensity'][i]
    state_atemp = segments_cool_humid['atemp_c'][i]
    state_rh   = segments_cool_humid['rh_pct'][i]

    # Store results
    state = (state_td, state_st, state_int, state_atemp, state_rh)

    pred = model.predict(np.array([state]))
    preds.append(state_st)

    print("State after segment", i+1, ":", pred)

#print("MSE:", mse, " R²:", r2)

# Convert predictions to a flat list
#predicted_skin_temp = [p[0] for p in preds]
predicted_skin_temp = preds

biogears_df = pd.read_csv(outputs_dir + '/biogears_results.csv')
#xml = segments_to_xml(segments_cool_humid)
#biogears_df = run_biogears(xml, segments_cool_humid)
#biogears_df.to_csv("outputs/biogears_results.csv")

print(biogears_df)
# Calculate mean absolute error
mae = mean_absolute_error(biogears_df['SkinTemperature(degC)'], predicted_skin_temp)
print("Mean Absolute Error:", mae)

import matplotlib.pyplot as plt

# Extract skin temperature from biogears results
biogears_skin_temp = biogears_df['SkinTemperature(degC)']



# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(list(biogears_skin_temp), label='BioGears Skin Temperature', color='blue')
plt.plot(list(predicted_skin_temp), label='Predicted Skin Temperature', color='red', linestyle='--')
plt.xlabel('Segment')
plt.ylabel('SkinTemperature(degC)')
plt.title('BioGears vs Predicted Skin Temperature')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(visualizations_dir, 'skin_temperature_comparison.png'))
plt.show()