from biogears_python.execution import generate_dynamic_intensity_synthetic_data
import time
import numpy as np # type: ignore

intensities = np.linspace(start=0.25, stop=1.0, num=10).tolist()
param_ranges = {
    'atemp_c':   np.linspace(start=20.0, stop=30.0, num=10).tolist(),
    'rh_pct':    np.linspace(start=40.0, stop=80.0, num=10).tolist(),
}

start_time = time.time()
max_workers = 10

generate_dynamic_intensity_synthetic_data(
    param_ranges=param_ranges,
    output_dir="simulation_results",  # CSV files will be placed in this folder
    intensities=intensities,
    n_iterations=10,
    time_per_segment=1.0,
    n_segments=10,
    show_progress=True,
    skip_existing=True,
    max_workers=max_workers
)

end_time = time.time()

print("Time elapsed:", end_time - start_time, "seconds")
print("Workers:", max_workers)