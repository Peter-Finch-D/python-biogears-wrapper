# scenario_model_example_plot.py
#
# A Python script showing how to use the ScenarioModel, then plot
# heart rate, core temperature, and skin temperature on separate
# subplots in Matplotlib.

import matplotlib.pyplot as plt

from scenario_model import ScenarioModel
from inputs1 import Inputs1
from inputs2 import Inputs2
from inputs3 import Inputs3

def main():
    # 1. Create an instance of the ScenarioModel (Python port).
    model = ScenarioModel()

    # 2. Prepare Inputs1, Inputs2, Inputs3 with example data
    inputs1 = Inputs1()
    inputs2 = Inputs2()
    inputs3 = Inputs3()

    # Populate Inputs1 (environment & subject)
    inputs1.setBW(70.0)       # Body weight in kg
    inputs1.setSA(1.8)        # Body surface area in m^2
    inputs1.setAGE(30)        # Subject age
    inputs1.setTa(30.0)
    inputs1.setTmr(30.0)
    inputs1.setVair(0.3)      # Ambient air velocity (m/s)
    inputs1.setPvap(15.0)     # Ambient vapor pressure (Torr)
    inputs1.setIclo(0.6)      # Clothing insulation
    inputs1.setIm(0.3)        # Clothing vapor permeability index
    inputs1.setPctFat(15.0)   # Percent body fat

    # Populate Inputs2 (metabolic rates & external work)
    inputs2.setWorkMode("f")    # 'f' = free walking mode
    inputs2.setVmove(1.2)       # Walking velocity (m/s)
    inputs2.setMtot(300.0)      # Total metabolic rate (W)
    inputs2.setMrst(70.0)       # Resting metabolic rate (W)
    inputs2.setMext(20.0)       # External workload (W)
    inputs2.setFluidIntake(0.5) # Fluid intake rate (g/min)

    # Populate Inputs3 (extended parameters)
    inputs3.setAcclimIndex(0)   # 0 = no acclimation
    inputs3.setDehydIndex(0)    # 0 = normal hydration
    inputs3.setStartTime(9)     # Start time = 9:00 AM
    inputs3.setTcr0(37.0)       # Override rectal temperature
    inputs3.setCircadianModel(False)
    inputs3.setTcoreOverride(False)

    try:
        # 3. Initialize the model
        model.init(inputs1, inputs2, inputs3)

        # 4. We'll collect the data for plotting
        times = []
        Tcr_values = []
        Tsk_values = []
        HR_values  = []

        currentTime = 0.0
        endTime = 60.0
        stepSize = 1.0

        while currentTime < endTime:
            nextTime = currentTime + stepSize
            if nextTime > endTime:
                nextTime = endTime

            # Step the model
            model.step(inputs1, inputs2, nextTime)

            # Retrieve the outputs
            p1 = model.getPredict1()
            p7 = model.getPredict7()
            p6 = model.getPredict6()

            # Print results (as in the original example)
            print(f"Time = {nextTime:.1f} min")
            print(f"  Rectal temperature (Tcr) = {p1.getTcr():.2f} degC")
            print(f"  Skin temperature (Tsk)   = {p1.getTsk():.2f} degC")
            print(f"  Heart Rate (HR)          = {p7.getHR():.2f} bpm")
            print(f"  Sweat Rate (SR)          = {p6.getSR():.2f} g/min")
            print(f"  Core-Skin dT (CrSkdT)    = {p6.getCrSkdT():.2f} degC")
            print(f"  PSI (Physiol. Strain)    = {p7.getPSI():.2f}")
            print()

            # Store data for plotting
            times.append(nextTime)
            Tcr_values.append(p1.getTcr())
            Tsk_values.append(p1.getTsk())
            HR_values.append(p7.getHR())

            currentTime = nextTime

        # 5. Clean up
        model.exit()

        # 6. Plot the results using Matplotlib
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Subplot for Rectal Temperature
        axs[0].plot(times, Tcr_values, color='red', label='Rectal Temperature')
        axs[0].set_ylabel('Tcr (degC)')
        axs[0].set_title('Core Temperature vs. Time')
        axs[0].legend(loc='best')
        axs[0].grid(True)

        # Subplot for Skin Temperature
        axs[1].plot(times, Tsk_values, color='blue', label='Skin Temperature')
        axs[1].set_ylabel('Tsk (degC)')
        axs[1].set_title('Skin Temperature vs. Time')
        axs[1].legend(loc='best')
        axs[1].grid(True)

        # Subplot for Heart Rate
        axs[2].plot(times, HR_values, color='green', label='Heart Rate')
        axs[2].set_ylabel('HR (bpm)')
        axs[2].set_xlabel('Time (min)')
        axs[2].set_title('Heart Rate vs. Time')
        axs[2].legend(loc='best')
        axs[2].grid(True)

        plt.tight_layout()

        # 7. Save the figure
        plt.savefig('scenario_results.png')
        plt.show()  # optional: to display the figure on-screen

    except Exception as e:
        print("An error occurred in the thermoregulatory model:", str(e))

if __name__ == "__main__":
    main()
