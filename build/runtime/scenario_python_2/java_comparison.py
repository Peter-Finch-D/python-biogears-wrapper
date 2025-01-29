# java_comparison_csv.py
#
# A Python script that replicates the ScenarioModelExample.java behavior,
# but prints time, rectal temperature, skin temperature, heart rate, sweat rate, etc.,
# in CSV format at each time step from 0..60 minutes in 1-minute increments.

from scenario_model import ScenarioModel
from inputs1 import Inputs1
from inputs2 import Inputs2
from inputs3 import Inputs3

def main():
    # 1. Create an instance of the ScenarioModel (Python port).
    model = ScenarioModel()

    # 2. Prepare Inputs1, Inputs2, Inputs3 with the same example data as in the Java code.

    inputs1 = Inputs1()
    inputs2 = Inputs2()
    inputs3 = Inputs3()

    # -----------------------------------------------------
    # Populate Inputs1 (environment & subject)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Populate Inputs2 (metabolic rates & external work)
    # -----------------------------------------------------
    inputs2.setWorkMode("f")   # 'f' = free walking mode
    inputs2.setVmove(1.2)      # Walking velocity (m/s)
    inputs2.setMtot(300.0)     # Total metabolic rate (W)
    inputs2.setMrst(70.0)      # Resting metabolic rate (W)
    inputs2.setMext(20.0)      # External workload (W)
    inputs2.setFluidIntake(0.5)  # Fluid intake rate (g/min)

    # -----------------------------------------------------
    # Populate Inputs3 (extended parameters)
    # -----------------------------------------------------
    inputs3.setAcclimIndex(0)   # 0 = no acclimation
    inputs3.setDehydIndex(0)    # 0 = normal hydration
    inputs3.setStartTime(9)     # Start time = 9:00 AM
    inputs3.setTcr0(37.0)       # Override rectal temperature
    inputs3.setCircadianModel(False)
    inputs3.setTcoreOverride(False)

    # 3. Initialize the model
    try:
        model.init(inputs1, inputs2, inputs3)

        # 4. Step the model from 0..60 min in 1-min increments.
        currentTime = 0.0
        endTime = 60.0
        stepSize = 1.0

        # Print CSV header
        print("Time,RectalTemp,SkinTemp,HeartRate,SweatRate,CoreSkin_dT,PSI")

        while currentTime < endTime:
            nextTime = currentTime + stepSize
            if nextTime > endTime:
                nextTime = endTime

            # Perform a simulation step up to nextTime
            model.step(inputs1, inputs2, nextTime)

            # Retrieve the outputs after this time step
            p1 = model.getPredict1()   # Core & Skin temps
            p6 = model.getPredict6()   # Sweat Rate, Core-Skin dT
            p7 = model.getPredict7()   # Heart Rate, PSI

            # Print a CSV row: time,Tcr,Tsk,HR,SR,CrSkdT,PSI
            print(f"{nextTime:.1f},"
                  f"{p1.getTcr():.2f},"
                  f"{p1.getTsk():.2f},"
                  f"{p7.getHR():.2f},"
                  f"{p6.getSR():.2f},"
                  f"{p6.getCrSkdT():.2f},"
                  f"{p7.getPSI():.2f}")

            currentTime = nextTime

        # 5. Clean up
        model.exit()

    except Exception as e:
        # If model raises an exception, mimic Java's catch block
        print("An error occurred in the thermoregulatory model:", str(e))

if __name__ == "__main__":
    main()
