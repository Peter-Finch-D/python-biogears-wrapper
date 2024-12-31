from biogears_python.xmlscenario import segments_to_xml # We use to build the Scenario XML
from biogears_python.execution import run_biogears      # We use to run the Scenario


# Create two different scenario dictionaries
segments_hot = { # Hot scenario
    'time'      : [10.00, 10.00, 10.00, 10.00, 10.00],
    'intensity' : [00.25, 00.25, 00.25, 00.25, 00.25],
    'atemp_c'   : [35.00, 35.00, 35.00, 35.00, 35.00],
    'rh_pct'    : [75.00, 75.00, 75.00, 75.00, 75.00],
}

segments_cold = { # Cold scenario
    'time'      : [10.00, 10.00, 10.00, 10.00, 10.00],
    'intensity' : [00.25, 00.25, 00.25, 00.25, 00.25],
    'atemp_c'   : [22.00, 22.00, 22.00, 22.00, 22.00],
    'rh_pct'    : [50.00, 50.00, 50.00, 50.00, 50.00],
}

xml_string_cold = segments_to_xml(segments_cold)
xml_string_hot  = segments_to_xml(segments_hot)
results_hot = run_biogears(xml_string_hot, segments_hot)
results_cold = run_biogears(xml_string_cold, segments_cold)

print("Hot scenario results:")
print(results_hot.head())
print("Cold scenario results:")
print(results_cold.head())

import matplotlib.pyplot as plt # type: ignore

plt.figure(figsize=(10, 6))
plt.plot(results_hot.index, results_hot['CoreTemperature(degC)'], label='Hot Environment')
plt.plot(results_cold.index, results_cold['CoreTemperature(degC)'], label='Cold Environment')
plt.grid(True)
plt.savefig('visualizations/core_temperature_comparison.png')