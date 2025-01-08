# Main.py

import pandas as pd
import numpy as np
import os
from digital_twin.data_processing import load_and_process_data
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from biogears_python.execution import run_biogears
from biogears_python.xmlscenario import segments_to_xml

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
    'intensity', 
    'atemp_c', 
    'rh_pct'
]
target_cols = [
    'SkinTemperature(degC)_next'
]

# Define initial state of model
initial_state = (
    1,
    0.25,
    22.0,
    90.0
)

# Define the exercise trial for the testing of the model run function
n_segments = 20
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
    next=True,
    time_deltas=True
)

# Prepare data
X = results['df'][feature_cols]
y = results['df'][target_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost with custom loss function
xg_reg = xgb.XGBRegressor(n_estimators=10000, max_depth=8, learning_rate=0.1, random_state=42, objective='reg:squarederror')
xg_reg.fit(X_train, y_train)

# Save model
model_save_path = 'outputs/models/xgb_model.json'
xg_reg.save_model(model_save_path)

# Evaluate model
y_pred = xg_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

pred = xg_reg.predict(np.array([initial_state]))

preds = []

state = initial_state
for i in range(len(segments_cool_humid['time'])):
    # Construct new state by adding the predicted deltas
    state_td  = state[0] + 1
    state_int = segments_cool_humid['intensity'][i]
    state_atemp = segments_cool_humid['atemp_c'][i]
    state_rh   = segments_cool_humid['rh_pct'][i]

    # Store results
    state = (state_td, state_int, state_atemp, state_rh)

    pred = xg_reg.predict(np.array([state]))
    preds.append(pred)

    print("State after segment", i+1, ":", pred)

#print("MSE:", mse, " R²:", r2)

#biogears_df = pd.read_csv(outputs_dir + '/bg_results.csv')
xml = segments_to_xml(segments_cool_humid)
biogears_df = run_biogears(xml, segments_cool_humid)
biogears_df.to_csv("outputs/biogears_results.csv")

print(biogears_df)

import matplotlib.pyplot as plt

# Extract skin temperature from biogears results
biogears_skin_temp = biogears_df['SkinTemperature(degC)']

# Convert predictions to a flat list
predicted_skin_temp = [p[0] for p in preds]

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