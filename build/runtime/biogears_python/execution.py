import itertools
import os
import io
import subprocess
import random
import torch  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Any, List, Tuple, Union
from tqdm import tqdm  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed

from biogears_python.xmlscenario import segments_to_xml

EXECUTABLE_LOCATION = './../outputs/bin/howto-runscenario'
XML_LOCATION = 'Scenarios/Dynamic_Scenario.xml'
RESULTS_LOCATION = 'HowTo-RunScenarioResults.csv'

def run_biogears(xml: str, segments: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Runs a single scenario string through BioGears and returns the CSV output as a DataFrame,
    assuming the BioGears executable prints CSV-formatted data to stdout.
    """
    segments_df = pd.DataFrame(segments)
    segments_df["Time(s)"] = (
        pd.to_timedelta(segments_df["time"], unit="m").cumsum()
        - pd.to_timedelta(segments_df["time"], unit="m")
    )
    segments_df.set_index("Time(s)", inplace=True)
    segments_df.drop(columns=["time"], inplace=True)

    result = subprocess.run(
        [EXECUTABLE_LOCATION, xml],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print("BioGears returned an error code:", result.returncode)
        print("Stderr:", result.stderr)
        raise RuntimeError("BioGears execution failed.")

    csv_data = result.stdout
    df = pd.read_csv(io.StringIO(csv_data))

    # Minimal columns rename for consistency
    df.columns = [
        'Time(s)',
        'HeartRate(1/min)',
        'CoreTemperature(degC)',
        'SkinTemperature(degC)'
    ]
    if "Time(s)" in df.columns:
        df["Time(s)"] = pd.to_timedelta(df["Time(s)"], unit="s")
        df.set_index("Time(s)", inplace=True)

    merged_df = pd.merge_asof(df, segments_df, left_index=True, right_index=True)
    # Resample merged_df to 1-minute intervals
    merged_df = merged_df.resample('1min').mean()

    return merged_df

def run_lstm_sequence(model, segments, scaler_X, scaler_Y, seq_length, initial_state=(0.0, 0.0, 0.0), device='cpu') -> pd.DataFrame:
    """
    Perform multi-step prediction using the trained LSTM model.
    Returns a pandas DataFrame similar to run_lstm in execution.py.
    
    Parameters:
    - model: Trained PyTorch model.
    - segments: Dictionary containing 'time', 'intensity', 'atemp_c', 'rh_pct'.
    - scaler_X: Fitted StandardScaler for input features.
    - scaler_Y: Fitted StandardScaler for target features.
    - seq_length: Number of timesteps in the input sequence.
    - initial_state: Tuple containing initial physiological states (HeartRate, CoreTemperature, SkinTemperature).
    - device: Device to run the model on ('cpu' or 'cuda').
    
    Returns:
    - df_predictions: pandas DataFrame with predictions and environmental parameters.
    """
    model.eval()
    times = segments.get('time', [])
    intensities = segments.get('intensity', [])
    atemps = segments.get('atemp_c', [])
    rhs = segments.get('rh_pct', [])
    n_steps = len(times)
    
    if n_steps == 0:
        raise ValueError("segments['time'] is empty; nothing to infer.")
    
    # Initialize records
    records = {
        "Time(s)": [], 
        "HeartRate(1/min)": [], 
        "CoreTemperature(degC)": [], 
        "SkinTemperature(degC)": [],
        "intensity": [], 
        "atemp_c": [], 
        "rh_pct": []
    }
    
    # Initial physiological states
    hr, ct, st = initial_state
    
    # Define start offset and time step
    start_offset = pd.to_timedelta("0 days 00:00:01.020000")
    one_min = pd.to_timedelta("00:01:00")
    
    # Initialize input sequence with initial physiological states and environmental factors
    input_seq = pd.DataFrame({
        'HeartRate(1/min)': [hr] * seq_length,
        'CoreTemperature(degC)': [ct] * seq_length,
        'SkinTemperature(degC)': [st] * seq_length,
        'intensity': [intensities[0]] * seq_length,
        'atemp_c': [atemps[0]] * seq_length,
        'rh_pct': [rhs[0]] * seq_length,
    })
    
    # Scale the input sequence
    input_scaled = scaler_X.transform(input_seq)
    
    for timestep in range(n_steps):
        # Current environmental parameters
        current_intensity = intensities[timestep]
        current_atemp = atemps[timestep]
        current_rh = rhs[timestep]
        
        # Convert to tensor
        X_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, T, F)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()[0]  # Shape: (P, 3)
        
        # Inverse transform the prediction
        pred = scaler_Y.inverse_transform(pred_scaled)  # Shape: (P, 3)
        
        # Use the last prediction for the state
        hr, ct, st = pred[-1]  # Shape: (3,)
        
        # Calculate current time
        current_time = start_offset + timestep * one_min
        
        # Record the last prediction and environmental parameters
        records["Time(s)"].append(current_time)
        records["HeartRate(1/min)"].append(hr)
        records["CoreTemperature(degC)"].append(ct)
        records["SkinTemperature(degC)"].append(st)
        records["intensity"].append(current_intensity)
        records["atemp_c"].append(current_atemp)
        records["rh_pct"].append(current_rh)
        
        # Prepare the new input entry with updated physiological states and current environmental factors
        new_entry = {
            'HeartRate(1/min)': hr,
            'CoreTemperature(degC)': ct,
            'SkinTemperature(degC)': st,
            'intensity': current_intensity,
            'atemp_c': current_atemp,
            'rh_pct': current_rh,
        }
        
        # Scale the new entry
        new_entry_scaled = scaler_X.transform(pd.DataFrame([new_entry]))
        
        # Update the input sequence by removing the oldest timestep and adding the new entry
        input_scaled = np.vstack([input_scaled[1:], new_entry_scaled])
    
    # Create DataFrame from records
    df_predictions = pd.DataFrame(records)
    df_predictions.set_index("Time(s)", inplace=True)
    
    return df_predictions

def run_lstm(
    model: torch.nn.Module,
    segments: dict,
    initial_state=(75.0, 37.0, 33.0),
    device: str = 'cpu',
    scaler=None
) -> pd.DataFrame:
    model.eval()
    times = segments.get('time', [])
    intensities = segments.get('intensity', [])
    atemps = segments.get('atemp_c', [])
    rhs = segments.get('rh_pct', [])
    n_steps = len(times)
    if n_steps == 0:
        raise ValueError("segments['time'] is empty; nothing to infer.")
    records = {
        "Time(s)": [], "HeartRate(1/min)": [], "CoreTemperature(degC)": [], "SkinTemperature(degC)": [],
        "intensity": [], "atemp_c": [], "rh_pct": []
    }
    hr, ct, st = initial_state
    start_offset = pd.to_timedelta("0 days 00:00:01.020000")
    one_min = pd.to_timedelta("00:01:00")
    if scaler:
        initial_df = pd.DataFrame({
            'HeartRate(1/min)': [hr], 'CoreTemperature(degC)': [ct], 'SkinTemperature(degC)': [st],
            'intensity': [0], 'atemp_c': [0], 'rh_pct': [0],
        })
        scaled_initial_df = pd.DataFrame(scaler.transform(initial_df), columns=initial_df.columns)
        hr = scaled_initial_df['HeartRate(1/min)'].iloc[0]
        ct = scaled_initial_df['CoreTemperature(degC)'].iloc[0]
        st = scaled_initial_df['SkinTemperature(degC)'].iloc[0]
    for i in range(n_steps):
        original_intensity = intensities[i]
        original_atemp = atemps[i]
        original_rh = rhs[i]
        if scaler:
            input_df = pd.DataFrame({
                'HeartRate(1/min)': [hr], 'CoreTemperature(degC)': [ct], 'SkinTemperature(degC)': [st],
                'intensity': [intensities[i]], 'atemp_c': [atemps[i]], 'rh_pct': [rhs[i]],
            })
            scaled_input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            intensity = scaled_input_df['intensity'].iloc[0]
            atemp = scaled_input_df['atemp_c'].iloc[0]
            rh = scaled_input_df['rh_pct'].iloc[0]
        else:
            intensity = intensities[i]
            atemp = atemps[i]
            rh = rhs[i]
        input_tensor = torch.tensor([[[hr, ct, st, intensity, atemp, rh]]], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.squeeze(dim=0).squeeze(dim=0)
        hr = pred[0].item()
        ct = pred[1].item()
        st = pred[2].item()
        if scaler:
            output_df = pd.DataFrame({
                'HeartRate(1/min)': [hr], 'CoreTemperature(degC)': [ct], 'SkinTemperature(degC)': [st],
                'intensity': [0], 'atemp_c': [0], 'rh_pct': [0],
            })
            descaled_output_df = pd.DataFrame(scaler.inverse_transform(output_df), columns=output_df.columns)
            hr = descaled_output_df['HeartRate(1/min)'].iloc[0]
            ct = descaled_output_df['CoreTemperature(degC)'].iloc[0]
            st = descaled_output_df['SkinTemperature(degC)'].iloc[0]
        intensity = original_intensity
        atemp = original_atemp
        rh = original_rh
        current_time = start_offset + i * one_min
        records["Time(s)"].append(current_time)
        records["HeartRate(1/min)"].append(hr)
        records["CoreTemperature(degC)"].append(ct)
        records["SkinTemperature(degC)"].append(st)
        records["intensity"].append(intensity)
        records["atemp_c"].append(atemp)
        records["rh_pct"].append(rh)
    df = pd.DataFrame(records)
    df['Time(s)'] = pd.to_timedelta(df['Time(s)'])
    df.set_index("Time(s)", inplace=True)
    return df

def generate_static_synthetic_data(
    param_ranges: Dict[str, Union[List[float], Tuple[float, float, float]]],
    output_dir: str,
    time_per_segment: float = 10.0,
    n_segments: int = 5,
    show_progress: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1
) -> None:
    """
    Generates multiple static BioGears scenarios by taking every combination of parameter values
    in 'param_ranges'. For each scenario, runs BioGears (via run_biogears) and saves the result
    to a .csv file named according to the parameter values, in 'output_dir'.

    Now supports concurrency via `max_workers`. Scenarios are queued until a thread is free.
    """
    os.makedirs(output_dir, exist_ok=True)

    expanded_values = _expand_param_values(param_ranges)
    param_keys = list(expanded_values.keys())
    all_combinations = list(itertools.product(*[expanded_values[k] for k in param_keys]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {}
        for combo in all_combinations:
            scenario_dict = {'time': [time_per_segment] * n_segments}
            for i, k in enumerate(param_keys):
                scenario_dict[k] = [combo[i]] * n_segments

            combo_str = "_".join([f"{k}={combo[i]}" for i, k in enumerate(param_keys)])
            filename = f"results_{combo_str}.csv"
            save_path = os.path.join(output_dir, filename)

            if skip_existing and os.path.isfile(save_path):
                continue

            xml_scenario = segments_to_xml(scenario_dict)
            fut = executor.submit(_run_and_save, xml_scenario, scenario_dict, save_path)
            future_to_filename[fut] = filename

        if show_progress:
            with tqdm(total=len(future_to_filename), desc="Scenarios Completed") as pbar:
                for fut in as_completed(future_to_filename):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Error in scenario {future_to_filename[fut]}: {e}")
                    finally:
                        pbar.update(1)
        else:
            for fut in as_completed(future_to_filename):
                fut.result()

def generate_dynamic_intensity_synthetic_data(
    param_ranges: Dict[str, Union[List[float], Tuple[float, float, float]]],
    output_dir: str,
    intensities: List[float],
    n_iterations: int,
    time_per_segment: float = 10.0,
    n_segments: int = 5,
    show_progress: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1
) -> None:
    """
    Similar to generate_static_synthetic_data, but:
      1) Only takes param ranges for atemp_c and rh_pct (or any environment factors).
      2) Loops over the combinations n_iterations times.
      3) For each segment, a random intensity is chosen from the provided intensities list.

    The rest of the logic (XML scenario creation, concurrency, saving .csv) is the same.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Expand param_ranges
    expanded_values = _expand_param_values(param_ranges)
    param_keys = list(expanded_values.keys())

    # Cartesian product of all param values (e.g., atemp_c x rh_pct)
    all_combinations = list(itertools.product(*[expanded_values[k] for k in param_keys]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {}

        # We loop over each combination n_iterations times
        # so we get repeated sets of scenarios with random intensities each time.
        for combo in all_combinations:
            for iteration in range(n_iterations):
                scenario_dict = {'time': [time_per_segment] * n_segments}
                
                # Assign param values (like atemp_c, rh_pct) to all segments
                for i, k in enumerate(param_keys):
                    scenario_dict[k] = [combo[i]] * n_segments

                # Random intensity for each segment
                scenario_dict['intensity'] = [
                    random.choice(intensities) for _ in range(n_segments)
                ]

                # Build filename that includes combo info + iteration index
                combo_str = "_".join([f"{k}={combo[i]}" for i, k in enumerate(param_keys)])
                filename = f"results_{combo_str}_iter={iteration}.csv"
                save_path = os.path.join(output_dir, filename)

                if skip_existing and os.path.isfile(save_path):
                    continue

                # Convert scenario dict to XML
                xml_scenario = segments_to_xml(scenario_dict)

                # Submit job
                fut = executor.submit(_run_and_save, xml_scenario, scenario_dict, save_path)
                future_to_filename[fut] = filename

        # Show progress if requested
        if show_progress:
            with tqdm(total=len(future_to_filename), desc="Dynamic Intensities") as pbar:
                for fut in as_completed(future_to_filename):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Error in scenario {future_to_filename[fut]}: {e}")
                    finally:
                        pbar.update(1)
        else:
            for fut in as_completed(future_to_filename):
                fut.result()

def _expand_param_values(
    param_ranges: Dict[str, Union[List[float], Tuple[float, float, float]]]
) -> Dict[str, List[float]]:
    """
    Expands parameter ranges if given as (start, stop, step) into discrete lists,
    or leaves them as-is if already lists.
    """
    expanded = {}
    for key, val in param_ranges.items():
        if isinstance(val, tuple) and len(val) == 3:
            start, stop, step = val
            values = np.arange(start, stop, step)
            expanded[key] = [round(v, 6) for v in values]  # Round to reduce float noise
        elif isinstance(val, list):
            expanded[key] = val
        else:
            raise ValueError(
                f"Parameter '{key}' must be either a list or a 3-tuple (start, stop, step)."
            )
    return expanded

def _run_and_save(xml_scenario: str, segments_dict, save_path: str) -> None:
    """
    Helper function that runs BioGears on 'xml_scenario' and saves the resulting DataFrame
    to 'save_path'. This is the function actually submitted to each worker thread.
    """
    df_output = run_biogears(xml_scenario, segments_dict)
    df_output.to_csv(save_path, index=True)
