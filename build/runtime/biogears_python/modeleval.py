import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from biogears_python.xmlscenario import segments_to_xml
from biogears_python.execution import run_biogears
from biogears_python.utils import print_with_timestamp
import matplotlib.pyplot as plt # type: ignore
import os
import torch # type: ignore

def evaluate_model_on_segments(segments, trained_model, visualize=False, verbose=False):
    
    # Ensure the segment durations are all equal to one
    for duration in segments['time']:
        if duration != 1:
            raise ValueError("Segment durations must all be equal to 1.")

    if verbose: print_with_timestamp("Generating BioGears ground truth data.")

    # Generate ground truth values from BioGears
    xml = segments_to_xml(segments)
    biogears_results = run_biogears(xml, segments)

    if verbose:
        print(biogears_results.head())
        print("Length of biogears results: ", len(biogears_results))

    if verbose: print_with_timestamp("Predicting with trained model.")
    trained_model.eval()  # Set model to evaluation mode

    # Initialize input features with the first row of BioGears results
    inputs = torch.tensor([[
        biogears_results['HeartRate(1/min)'].iloc[0],
        biogears_results['CoreTemperature(degC)'].iloc[0],
        biogears_results['SkinTemperature(degC)'].iloc[0],
        segments['intensity'][0],
        segments['atemp_c'][0],
        segments['rh_pct'][0]
    ]], dtype=torch.float32)

    # Prepare containers for storing predictions
    model_results = {
        'Time(s)': [],
        'HeartRate(1/min)': [],
        'CoreTemperature(degC)': [],
        'SkinTemperature(degC)': []
    }

    with torch.no_grad():
        for i in range(len(segments['time'])):
            # Add time to results
            model_results['Time(s)'].append(i)
            
            # Make prediction
            prediction = trained_model(inputs.unsqueeze(0))  # Add batch dimension
            predicted_values = prediction.squeeze(0).numpy()

            # Append results
            model_results['HeartRate(1/min)'].append(predicted_values[0])
            model_results['CoreTemperature(degC)'].append(predicted_values[1])
            model_results['SkinTemperature(degC)'].append(predicted_values[2])

            # Update input features for the next time step
            if i + 1 < len(segments['time']):
                next_inputs = [
                    predicted_values[0],
                    predicted_values[1],
                    predicted_values[2],
                    segments['intensity'][i + 1],
                    segments['atemp_c'][i + 1],
                    segments['rh_pct'][i + 1]
                ]
                inputs = torch.tensor([next_inputs], dtype=torch.float32)

    # Convert results to DataFrame
    model_results_df = pd.DataFrame(model_results)
    model_results_df["Time(s)"] = pd.to_timedelta(model_results_df["Time(s)"], unit="m")
    model_results_df.set_index("Time(s)", inplace=True)

    # Calculate MAE
    mae_hr = mean_absolute_error(biogears_results['HeartRate(1/min)'], model_results_df['HeartRate(1/min)'])
    mae_ct = mean_absolute_error(biogears_results['CoreTemperature(degC)'], model_results_df['CoreTemperature(degC)'])
    mae_st = mean_absolute_error(biogears_results['SkinTemperature(degC)'], model_results_df['SkinTemperature(degC)'])

    print_with_timestamp(f"Mean Absolute Error - Heart Rate: {mae_hr:.2f}, Core Temperature: {mae_ct:.2f}, Skin Temperature: {mae_st:.2f}")

    # Visualization
    if visualize:
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        axs[0].plot(biogears_results['HeartRate(1/min)'], label='BioGears Heart Rate')
        axs[0].plot(model_results_df['HeartRate(1/min)'], label='Predicted Heart Rate')
        axs[0].set_title('Heart Rate Prediction')
        axs[0].legend()

        axs[1].plot(biogears_results['CoreTemperature(degC)'], label='BioGears Core Temperature')
        axs[1].plot(model_results_df['CoreTemperature(degC)'], label='Predicted Core Temperature')
        axs[1].set_title('Core Temperature Prediction')
        axs[1].legend()

        axs[2].plot(biogears_results['SkinTemperature(degC)'], label='BioGears Skin Temperature')
        axs[2].plot(model_results_df['SkinTemperature(degC)'], label='Predicted Skin Temperature')
        axs[2].set_title('Skin Temperature Prediction')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig('visualizations/model_evaluation.png')

