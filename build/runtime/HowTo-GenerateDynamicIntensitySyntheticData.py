from biogears_python.execution import generate_dynamic_intensity_synthetic_data
import time
import numpy as np # type: ignore

intensities = np.linspace(start=0.25, stop=1.0, num=10).tolist()
param_ranges = {
    'atemp_c':   np.linspace(start=10.0, stop=30.0, num=7).tolist(),
    'rh_pct':    np.linspace(start=0.0, stop=100.0, num=7).tolist(),
}

start_time = time.time()
max_workers = 10

generate_dynamic_intensity_synthetic_data(
    param_ranges=param_ranges,
    output_dir="simulation_results",  # CSV files will be placed in this folder
    intensities=intensities,
    n_iterations=2,
    time_per_segment=1.0,
    n_segments=60,
    show_progress=True,
    skip_existing=False,
    max_workers=max_workers
)

end_time = time.time()

print("Time elapsed:", end_time - start_time, "seconds")
print("Workers:", max_workers)