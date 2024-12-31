import itertools
import os
import io
import subprocess
import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Dict, Any, List, Tuple, Union
from tqdm import tqdm # type: ignore
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

    # Convert segments to DataFrame
    segments_df = pd.DataFrame(segments)
    segments_df["Time(s)"] = pd.to_timedelta(segments_df["time"], unit="m").cumsum() - pd.to_timedelta(segments_df["time"], unit="m")
    segments_df.set_index("Time(s)", inplace=True)
    segments_df.drop(columns=["time"], inplace=True)

    #print(segments_df)

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
    
    # Rename columns if desired (example)
    df.columns = [
        'Time(s)',
        'HeartRate(1/min)',
        'CardiacOutput(mL/min)',
        'MeanArterialPressure(mmHg)',
        'SystolicArterialPressure(mmHg)',
        'DiastolicArterialPressure(mmHg)',
        'TotalMetabolicRate(kcal/day)',
        'CoreTemperature(degC)',
        'SkinTemperature(degC)',
        'RespirationRate(1/min)',
        'AchievedExerciseLevel',
        'FatigueLevel',
        'TotalMetabolicRate(W)',
        'TotalWorkRateLevel'
    ]
    if "Time(s)" in df.columns:
        df["Time(s)"] = pd.to_timedelta(df["Time(s)"], unit="s")
        df.set_index("Time(s)", inplace=True)

    # Merge the segments DataFrame with the output DataFrame
    merged_df = pd.merge_asof(df, segments_df, left_index=True, right_index=True)

    return merged_df

def _expand_param_values(
    param_ranges: Dict[str, Union[List[float], Tuple[float, float, float]]]
) -> Dict[str, List[float]]:
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

    :param param_ranges:   Dict describing each parameter's possible values.
    :param output_dir:     Path to directory for .csv files.
    :param time_per_segment: Duration (minutes) of each segment.
    :param n_segments:     Number of consecutive segments per scenario.
    :param show_progress:  If True, show a tqdm progress bar.
    :param skip_existing:  If True, skip running scenarios for which CSV already exists.
    :param max_workers:    Maximum number of worker threads to use in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Expand any (start, stop, step) into discrete lists
    expanded_values = _expand_param_values(param_ranges)
    param_keys = list(expanded_values.keys())

    # 2) Cartesian product of all param values
    all_combinations = list(itertools.product(*[expanded_values[k] for k in param_keys]))

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {}

        for combo in all_combinations:
            scenario_dict = {'time': [time_per_segment] * n_segments}
            for i, k in enumerate(param_keys):
                scenario_dict[k] = [combo[i]] * n_segments

            # Construct a filename that includes all parameters
            combo_str = "_".join([f"{k}={combo[i]}" for i, k in enumerate(param_keys)])
            filename = f"results_{combo_str}.csv"
            save_path = os.path.join(output_dir, filename)

            # Skip if file already exists and skip_existing is True
            if skip_existing and os.path.isfile(save_path):
                continue

            # Convert scenario dict to XML and submit a job to the thread pool
            xml_scenario = segments_to_xml(scenario_dict)

            fut = executor.submit(_run_and_save, xml_scenario, scenario_dict, save_path)
            future_to_filename[fut] = filename

        # Show progress only for completed scenarios
        if show_progress:
            with tqdm(total=len(future_to_filename), desc="Scenarios Completed") as pbar:
                for fut in as_completed(future_to_filename):
                    try:
                        fut.result()  # Check for exceptions
                    except Exception as e:
                        print(f"Error in scenario {future_to_filename[fut]}: {e}")
                    finally:
                        pbar.update(1)
        else:
            # If not showing a progress bar, still raise any exceptions
            for fut in as_completed(future_to_filename):
                fut.result()  # Will raise if there's an error


def _run_and_save(xml_scenario: str, segments_dict, save_path: str) -> None:
    """
    Helper function that runs BioGears on 'xml_scenario' and saves the resulting DataFrame
    to 'save_path'. This is the function actually submitted to each worker thread.
    """
    df_output = run_biogears(xml_scenario, segments_dict)
    df_output.to_csv(save_path, index=True)
