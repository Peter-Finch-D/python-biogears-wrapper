"""
scenario_model.py

A Python module that implements the logic from ScenarioModel.java as closely
as possible, using Python, scenario_constants, tissue_conductance, and utils.

The old 2-step integration scheme in step() is replaced with scipy.integrate.solve_ivp.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from scenario_constants import (
    # Constants:
    NO_ACCLIMATION, PART_ACCLIMATION, FULL_ACCLIMATION,
    DEHYD_NORMAL, DEHYD_MODERATE, DEHYD_SEVERE,
    fMrstCr, fMrstMu, fMrstFat, fMrstVsk, fMrstSk,
    fCOrstCr, fCOrstMu, fCOrstFat, fCOrstVsk,
    SpHtCr, SpHtMu, SpHtFat, SpHtSk, SpHtBl,
    Lewis, Boltz, SArad, KClo, HtVap
)
from tissue_conductance import TissueConductance
import utils  # for SatVP, LagIt, etc.

# We'll define "Predict" as a simple container for final outputs
class Predict:
    def __init__(self):
        self.Tra  = 0.0
        self.Tcr  = 0.0
        self.Tmu  = 0.0
        self.Tfat = 0.0
        self.Tvsk = 0.0
        self.Tsk  = 0.0
        self.Tcl  = 0.0
        self.Tbdy = 0.0

        self.BFra  = 0.0
        self.BFcr  = 0.0
        self.BFmu  = 0.0
        self.BFfat = 0.0
        self.BFvsk = 0.0

        self.dQtot_dt = 0.0
        self.Qtot     = 0.0
        self.Qra      = 0.0
        self.Qcr      = 0.0
        self.Qmu      = 0.0
        self.Qfat     = 0.0
        self.Qvsk     = 0.0
        self.Qsk      = 0.0

        self.SR    = 0.0
        self.Esk   = 0.0
        self.Emax  = 0.0
        self.SkWet = 0.0
        self.dMshiv= 0.0
        self.MRC   = 0.0
        self.CrSkdT= 0.0
        self.RaSkdT= 0.0

        self.FluidLoss= 0.0
        self.O2debt   = 0.0
        self.HR       = 0.0
        self.SV       = 0.0
        self.PSI      = 0.0

        self.ThypSet  = 0.0
        self.TskSet   = 0.0
        self.TcrFlag  = 0.0
        self.HRFlag   = 0.0
        self.dQ_dtFlag= 0.0
        self.QtotFlag = 0.0
        self.TimeOfDay= 0.0

class Inputs1:
    """
    Mirrors the Java Inputs1 class, storing environment and subject data.
    """
    def __init__(self, BW=70.0, SA=1.8, AGE=35, Ta=25.0, Tmr=25.0,
                 Vair=0.5, Pvap=15.0, Iclo=0.5, Im=1.0, PctFat=15.0):
        self.BW     = BW
        self.SA     = SA
        self.AGE    = AGE
        self.Ta     = Ta
        self.Tmr    = Tmr
        self.Vair   = Vair
        self.Pvap   = Pvap
        self.Iclo   = Iclo
        self.Im     = Im
        self.PctFat = PctFat

class Inputs2:
    """
    Mirrors Java Inputs2, storing the metabolic/work data.
    """
    def __init__(self, Mtot=200.0, Mrst=100.0, Mext=20.0, Vmove=1.0,
                 workMode='r', fluidIntake=0.0):
        self.Mtot        = Mtot
        self.Mrst        = Mrst
        self.Mext        = Mext
        self.Vmove       = Vmove
        self.workMode    = workMode  # e.g. 'r', 't', 'f', etc.
        self.fluidIntake = fluidIntake

class Inputs3:
    """
    Mirrors Java Inputs3, storing acclimation, dehydration, circadian, etc.
    """
    def __init__(self,
                 acclimIndex=NO_ACCLIMATION,
                 dehydIndex=DEHYD_NORMAL,
                 startTime=0.0,
                 Tcr0=37.0,
                 circadianModel=False,
                 tcoreOverride=False):
        self.acclimIndex    = acclimIndex
        self.dehydIndex     = dehydIndex
        self.startTime      = startTime
        self.Tcr0           = Tcr0
        self.circadianModel = circadianModel
        self.tcoreOverride  = tcoreOverride

class ModelException(Exception):
    """
    A custom exception to mirror the Java ScenarioModel's ModelException.
    """
    pass

##############################################################################
# ODE SYSTEM for solve_ivp
##############################################################################

def _scenario_odes(t, y, model, inputs1, inputs2):
    """
    ODE function for ScenarioModel used by solve_ivp.

    We handle the state variables:
      y[0] = Tra
      y[1] = Tcr
      y[2] = Tmu
      y[3] = Tfat
      y[4] = Tvsk
      y[5] = Tsk
      y[6] = TTC
      y[7] = O2debt
      y[8] = fluidLoss

    We then compute derivatives based on model's conduction, vasomotion, etc.
    """
    # Assign to model so all the "comp" methods can see current T's
    model.Tra, model.Tcr, model.Tmu, model.Tfat, \
        model.Tvsk, model.Tsk, model.TTC, model.O2debt, model.fluidLoss = y

    # Also update model.time = t (minutes), to keep it consistent
    model.time = t

    # For clarity, rename frequently used environment/work variables:
    Ta   = inputs1.Ta
    Tmr  = inputs1.Tmr
    Pvap = inputs1.Pvap
    Vair = inputs1.Vair
    Iclo = inputs1.Iclo
    Im   = inputs1.Im

    workMode   = inputs2.workMode
    Vmove      = inputs2.Vmove
    fluidIntake= inputs2.fluidIntake

    # Also the model's Mtot, Mrst, Mext
    model.Mtot  = inputs2.Mtot
    Mrst        = inputs2.Mrst
    Mext        = inputs2.Mext
    model.Mnet  = model.Mtot - Mext

    # For muscle metabolism:
    Mmu = model.Mnet - (Mrst*(1.0 - fMrstMu))

    # VO2 => to determine BFvskMax, stroke volume, etc.
    VO2 = model.Mtot / 341.0
    if VO2 <= 0.5:
        SVmax = 85.0
        model.BFvskMaxNew = 7000.0
    elif VO2 >= 2.0:
        SVmax = 130.0
        model.BFvskMaxNew = 5000.0
    else:
        SVmax = 30.0*(VO2 - 0.5) + 85.0
        model.BFvskMaxNew = 7000.0 - (VO2 - 0.5)*1333.0

    # Eres, Cres
    Eres = model.SA * 0.0023 * model.Mnet * (44.0 - Pvap)
    Cres = model.SA * 0.0012 * model.Mnet * (34.0 - Ta)

    # Compute conduction/radiation coefficients:
    Kc = model.compKc(workMode, Vair, Vmove, Mrst)
    Kr = model.compKr(Kc, Iclo, Tmr, Ta)
    Kop = Kc + Kr
    model.Top = (Kr*Tmr + Kc*Ta)/Kop
    model.ClEffFact = 1.0/(1.0 + Kop*KClo*Iclo)

    # T chain
    model.compShiverCorrection(Mmu, model.Mtot, Mext, model.EEmu, (Mmu + Mext))
    model.compStrokeVolume(SVmax)
    model.compCOreq()
    model.compVascularBloodFlow(VO2)
    model.compVskSkCond(workMode)
    model.compMuscleBloodFlow()
    model.compShivering()
    model.compSweatRate()
    model.compDripSkWetEsk(Kop, Kc, Pvap, model.ClEffFact, Im)
    model.compShiverCorrection(Mmu, model.Mtot, Mext, model.EEmu, (Mmu + Mext))
    # Dry heat loss updates
    model.compDryHeatLoss(1.0 + 0.15*Iclo, Kc, Tmr, Ta, Iclo)

    # Evaluate derivatives using model.deriv => sets dQxxx_dt etc.
    model.deriv(Cres, Eres, model.MrstCr, model.MrstFat, model.MrstVsk, model.MrstSk)

    # Convert those heat flow derivatives into T derivatives
    dTra_dt  = model.dQra_dt  / model.HtCapRA
    dTcr_dt  = model.dQcr_dt  / model.HtCapCr
    dTmu_dt  = model.dQmu_dt  / model.HtCapMu
    dTfat_dt = model.dQfat_dt / model.HtCapFat
    dTvsk_dt = model.dQvsk_dt / model.HtCapVsk
    dTsk_dt  = model.dQsk_dt  / model.HtCapSk
    dTTC_dt  = model.dQTC_dt  / model.HtCapSk

    # O2 debt, fluidLoss
    dO2debt_dt    = model.dO2debt_dt
    dFluidLoss_dt = model.SR - fluidIntake

    return np.array([
        dTra_dt, dTcr_dt, dTmu_dt, dTfat_dt,
        dTvsk_dt, dTsk_dt, dTTC_dt,
        dO2debt_dt, dFluidLoss_dt
    ], dtype=float)

def _integrate_segment(model, inputs1, inputs2, t_start, t_end):
    """
    Replaces the old 'while self.time < Tnext' loop with a solve_ivp approach
    that integrates from t_start to t_end in one shot.
    """
    y0 = np.array([
        model.Tra, model.Tcr, model.Tmu, model.Tfat,
        model.Tvsk, model.Tsk, model.TTC,
        model.O2debt, model.fluidLoss
    ], dtype=float)

    sol = solve_ivp(
        fun=_scenario_odes,
        t_span=(t_start, t_end),
        y0=y0,
        args=(model, inputs1, inputs2),
        max_step=0.5,   # or smaller if desired
        rtol=1e-3,
        atol=1e-4
    )
    # The final state is sol.y[:, -1]
    y_final = sol.y[:, -1]
    (
        model.Tra, model.Tcr, model.Tmu, model.Tfat,
        model.Tvsk, model.Tsk, model.TTC,
        model.O2debt, model.fluidLoss
    ) = y_final

    model.time = t_end

    # Optionally log states at each step:
    # for i in range(sol.y.shape[1]):
    #     # update model from sol.y[:, i], then model.log_state()
    # We just log once at the end here:
    model.log_state()

class ScenarioModel:

    # Additional constants from Java
    CIave   = 3200.0
    HRrst   = 70.0
    KR0     = 4.7
    STANDARD_KC = 2.1
    BlO2Cap = 0.21  # cc O2 per cc blood
    O2Equiv = 2.93  # cc O2/(W-min)
    BFvskMaxOld = 7000.0
    BFvskMin    = 27.0
    TsetBFold   = 36.75
    SRmax       = 25.0
    SRmin       = 0.0
    DELTA_ALPHA_SR = -0.6
    DELTA_HYPSIG   = 0.06
    STANDARD_THYP_SET = 36.96
    STANDARD_TSK_SET  = 33.0
    STANDARD_TCR_FLAG = 39.5
    STANDARD_DQDT_FLAG= 95.0
    STANDARD_QTOT_FLAG= 840.0

    FAILED_TO_CONVERGE = "Scenario.compKr failed to converge."
    NEGATIVE_TIMESTEP  = "Negative time step computed.  Reduce the stability factor."

    def __init__(self, segments):
        """
        In this constructor, we store 'segments' of environment input,
        then initialize once, and integrate each segment with solve_ivp.
        """
        self.predict = Predict()

        # For storing states over time
        self.states = {k: [] for k in [
            "time","Tra","Tcr","Tmu","Tfat","Tvsk","Tsk","TTC",
            "BFra","BFcr","BFmu","BFfat","BFvsk",
            "HR","SV","CO","CI","SR","Esk","Emax","SkWet","dMshiv","fluidLoss",
            "O2debt","PSI","Qra","Qcr","Qmu","Qfat","Qvsk","Qsk","Qtot","Tbdy",
            "Overload","PctExt","dQra_dt","dQcr_dt","dQmu_dt","dQfat_dt",
            "dQvsk_dt","dQsk_dt","dQTC_dt"
        ]}

        # Time-independent placeholders
        self._init_model_vars()

        # Build the Inputs from segments
        def get_inputs_from_segments(segments, index):
            i1 = Inputs1(
                BW=segments['BW'][index],
                SA=segments['SA'][index],
                AGE=segments['AGE'][index],
                Ta=segments['Ta'][index],
                Tmr=segments['Tmr'][index],
                Vair=segments['Vair'][index],
                Pvap=segments['Pvap'][index],
                Iclo=segments['Iclo'][index],
                Im=segments['Im'][index],
                PctFat=segments['PctFat'][index]
            )
            i2 = Inputs2(
                Mtot=segments['Mtot'][index],
                Mrst=segments['Mrst'][index],
                Mext=segments['Mext'][index],
                Vmove=segments['Vmove'][index],
                workMode=segments['workMode'][index],
                fluidIntake=segments['fluidIntake'][index]
            )
            i3 = Inputs3(
                acclimIndex=segments['acclimIndex'][index],
                dehydIndex=segments['dehydIndex'][index],
                startTime=segments['startTime'][index],
                Tcr0=segments['Tcr0'][index],
                circadianModel=segments['circadianModel'][index],
                tcoreOverride=segments['tcoreOverride'][index]
            )
            return i1, i2, i3

        # Initialize with the first segment
        i1, i2, i3 = get_inputs_from_segments(segments, 0)
        self.init(i1, i2, i3)

        # Then integrate each segment
        T_next = 0.0
        for i in range(len(segments['time'])):
            self.log_state()
            i1, i2, i3 = get_inputs_from_segments(segments, i)
            dt = segments['time'][i]
            T_start = self.time
            T_next  = self.time + dt

            # Use solve_ivp from T_start to T_next
            _integrate_segment(self, i1, i2, T_start, T_next)

            # After finishing that segment, do final step logic if desired
            self._finish_segment(i1, i2, i3)
        self.log_state()

    def _init_model_vars(self):
        """
        Just sets placeholders for all relevant model variables to zero.
        """
        self.BW   = 0.0
        self.SA   = 0.0
        self.PctFat = 0.0

        self.MrstCr = 0.0
        self.MrstMu = 0.0
        self.MrstFat= 0.0
        self.MrstVsk= 0.0
        self.MrstSk = 0.0
        self.COrst  = 0.0
        self.BFcrOld= 0.0
        self.SVold  = 0.0
        self.HRmax  = 0.0

        self.HtCapRA = 0.0
        self.HtCapCr = 0.0
        self.HtCapMu = 0.0
        self.HtCapFat= 0.0
        self.HtCapVsk= 0.0
        self.HtCapSk = 0.0

        self.ThypSet = 0.0
        self.TskSet  = 0.0

        self.CrMuCond   = 0.0
        self.MuFatCond  = 0.0
        self.FatVskCond = 0.0
        self.VskSkCond  = 0.0
        self.SkCondConst= 0.0

        self.Tra  = 0.0
        self.Tcr  = 0.0
        self.Tmu  = 0.0
        self.Tfat = 0.0
        self.Tvsk = 0.0
        self.Tsk  = 0.0
        self.Tcl  = 0.0
        self.TTC  = 0.0

        self.BFra = 0.0
        self.BFcr = 0.0
        self.BFmu = 0.0
        self.BFfat= 0.0
        self.BFvsk= 0.0

        self.Qra  = 0.0
        self.Qcr  = 0.0
        self.Qmu  = 0.0
        self.Qfat = 0.0
        self.Qvsk = 0.0
        self.Qsk  = 0.0
        self.Qtot = 0.0

        self.SR   = 0.0
        self.Esk  = 0.0
        self.Emax = 0.0
        self.dMshiv  = 0.0
        self.fluidLoss=0.0
        self.O2debt   = 0.0
        self.TTC      = 0.0
        self.BFvskReq = 0.0
        self.BFmuReq  = 0.0
        self.BFvskMaxNew = 0.0
        self.EEmu     = 0.0
        self.SV       = 0.0
        self.HR       = 0.0
        self.PSI      = 0.0
        self.Tbdy     = 0.0

        self.Overload= 0.0
        self.PctHRmax= 0.0
        self.LeftOver= 0.0
        self.PctExt  = 80.0

        self.Mtot = 0.0
        self.Mmu  = 0.0
        self.Mnet = 0.0

        self.Top = 0.0
        self.ClEffFact = 0.0
        self.DryHtLoss1 = 0.0
        self.DryHtLoss2 = 0.0
        self.SkWet = 0.0
        self.Drip  = 0.0

        self.dQra_dt   = 0.0
        self.dQcr_dt   = 0.0
        self.dQmu_dt   = 0.0
        self.dQfat_dt  = 0.0
        self.dQvsk_dt  = 0.0
        self.dQsk_dt   = 0.0
        self.dQTC_dt   = 0.0
        self.dO2debt_dt= 0.0

        self.delThypSet   = 0.0
        self.acclimIndex  = NO_ACCLIMATION
        self.dehydIndex   = DEHYD_NORMAL
        self.startTime    = 0.0
        self.Tcr0         = 37.0
        self.circadianModel = False
        self.tcoreOverride  = False
        self.Tre0  = 0.0
        self.HR0   = 0.0

        self.dt   = 0.025
        self.FUZZ = 0.00025
        self.time = 0.0
        self.T_EEmuNew = 0.0
        self.iter = 0
        self.DEBUG = False
        self.stabilityFactor = 0.5

    def getPredict(self):
        return self.predict

    def init(self, inputs1, inputs2, inputs3):
        """
        Setup initial node temperatures, resting flows, etc.
        """
        # This is the same as your original 'init(...)'
        # (Truncated for brevity, but keep all the code from your original init.)

        # 1) read BW, SA, etc.
        self.BW   = inputs1.BW
        self.SA   = inputs1.SA
        age       = inputs1.AGE
        Ta        = inputs1.Ta
        Tmr       = inputs1.Tmr
        Vair      = inputs1.Vair
        Pvap      = inputs1.Pvap
        Iclo      = inputs1.Iclo
        Im        = inputs1.Im
        self.PctFat= inputs1.PctFat

        self.Mtot = inputs2.Mtot
        Mrst      = inputs2.Mrst
        Mext      = inputs2.Mext
        self.Mnet = self.Mtot - Mext
        workMode  = inputs2.workMode

        self.acclimIndex  = inputs3.acclimIndex
        self.dehydIndex   = inputs3.dehydIndex
        self.startTime    = inputs3.startTime
        self.Tcr0         = inputs3.Tcr0
        self.circadianModel = inputs3.circadianModel
        self.tcoreOverride  = inputs3.tcoreOverride

        # circadian stub
        def circadianInterpTime(t):
            return 36.96
        self.ThypSet = self.STANDARD_THYP_SET
        if self.circadianModel:
            self.ThypSet = circadianInterpTime(self.startTime)

        # Set node temps based on acclimation
        if self.acclimIndex == NO_ACCLIMATION:
            self.delThypSet = 0.0
            self.Tra, self.Tmu, self.Tcr = 36.75, 36.07, 36.98
            self.Tfat, self.Tvsk, self.Tsk = 33.92, 33.49, 33.12
            self.TTC = self.Tsk
        elif self.acclimIndex == PART_ACCLIMATION:
            self.delThypSet = -0.25
            self.Tra, self.Tmu, self.Tcr = 36.73, 36.10, 36.96
            self.Tfat, self.Tvsk, self.Tsk = 34.13, 33.73, 32.87
            self.TTC = self.Tsk
        elif self.acclimIndex == FULL_ACCLIMATION:
            self.delThypSet = -0.5
            self.Tra, self.Tmu, self.Tcr = 36.49, 35.91, 36.72
            self.Tfat, self.Tvsk, self.Tsk = 34.04, 33.66, 32.79
            self.TTC = self.Tsk

        self.ThypSet += self.delThypSet

        if self.tcoreOverride:
            delta = (self.Tcr0 - self.Tcr)
            self.Tra  += delta
            self.Tmu  += delta
            self.Tfat += delta
            self.Tvsk += delta
            self.Tsk  += delta
            self.Tcr   = self.Tcr0
            self.TTC   = self.Tsk

        # Dehydration
        if self.dehydIndex == DEHYD_NORMAL:
            self.fluidLoss = 0.0
        elif self.dehydIndex == DEHYD_MODERATE:
            self.fluidLoss = 0.02 * self.BW * 1000.0
        elif self.dehydIndex == DEHYD_SEVERE:
            self.fluidLoss = 0.04 * self.BW * 1000.0

        # Distribution of resting metabolism
        self.MrstCr  = fMrstCr  * Mrst
        self.MrstMu  = fMrstMu  * Mrst
        self.MrstFat = fMrstFat * Mrst
        self.MrstVsk = fMrstVsk * Mrst
        self.MrstSk  = fMrstSk  * Mrst

        # Blood flow distribution
        self.COrst = self.CIave * self.SA
        self.BFra  = self.COrst
        self.BFcr  = fCOrstCr  * self.COrst
        self.BFcrOld= self.BFcr
        self.BFmu  = fCOrstMu  * self.COrst
        self.BFfat = fCOrstFat * self.COrst
        self.BFvsk = fCOrstVsk * self.COrst
        self.BFvskReq = self.BFvsk
        self.BFmuReq  = self.BFmu

        # Tissue conductances
        tissueConductance = TissueConductance()
        tissueConductance.compute(inputs1)
        self.CrMuCond   = tissueConductance.getCrMuCond()
        self.MuFatCond  = tissueConductance.getMuFatCond()
        self.FatVskCond = tissueConductance.getFatVskCond()
        self.VskSkCond  = tissueConductance.getVskSkCond()
        self.SkCondConst= tissueConductance.getSkCondConst()

        Wfat  = 10.0 * self.PctFat * self.BW
        Wnfat = 10.0 * (100.0 - self.PctFat) * self.BW

        # Heat capacities
        self.HtCapRA = SpHtBl * 0.02 * Wnfat / 1000.0
        self.HtCapCr = SpHtCr * 0.37 * Wnfat / 1000.0
        self.HtCapMu = SpHtMu * 0.54 * Wnfat / 1000.0
        self.HtCapFat= SpHtFat* Wfat / 1000.0
        self.HtCapVsk= SpHtBl * 0.01 * Wnfat / 1000.0
        self.HtCapSk = SpHtSk * 0.06 * Wnfat / 1000.0

        # initial dryness
        self.SR = 0.0
        self.Esk= 0.0
        self.dMshiv= 0.0
        self.EEmu = self.MrstMu + (self.Mtot - Mrst)  # or similar

        SVmax = self.COrst / self.HRrst
        self.SVold = SVmax
        MWST = self.Tsk
        if MWST > 38: 
            MWST = 38
        if MWST <= 33:
            self.SV = self.SVold
        else:
            self.SV = self.SVold - 5.0*((self.SVold-85.0)/45.0)*(MWST - 33.0)

        COreq = self.BFcr + self.BFmuReq + self.BFfat + self.BFvskReq
        self.HR = COreq / self.SV

        self.BFvskMaxNew = self.BFvskMaxOld
        self.HR0  = self.HR
        self.Tre0 = self.Tcr

        self.HRmax = 220.0 - age
        self.dt   = 0.025
        self.FUZZ = 0.01*self.dt
        self.time = 0.0

    def _finish_segment(self, inputs1, inputs2, inputs3):
        """
        Optionally, do post-segment computations like final PSI, etc.
        """
        # Usually you'd do logic at the end of your old step method:
        self.compTbdy()
        CrSkdT = self.Tcr - self.Tsk
        RaSkdT = self.Tra - self.Tsk

        # For PSI:
        self.PSI = 5.0 * (
            (self.Tcr - self.Tre0)/(39.5 - self.Tre0)
            + (self.HR - self.HR0)/(180.0 - self.HR0)
        )
        if self.PSI < 0.0:
            self.PSI = 0.0

        MRC = self.Mnet  # or some variation that includes Eres, Cres
        dQtot_dt = (
            self.dQra_dt + self.dQcr_dt + self.dQmu_dt
            + self.dQfat_dt + self.dQvsk_dt + self.dQsk_dt
        )

        # Store final in self.predict if you want
        self.predict.Tra  = self.Tra
        self.predict.Tcr  = self.Tcr
        self.predict.Tmu  = self.Tmu
        self.predict.Tfat = self.Tfat
        self.predict.Tvsk = self.Tvsk
        self.predict.Tsk  = self.Tsk
        self.predict.Tbdy = self.Tbdy

        self.predict.BFra  = self.BFra
        self.predict.BFcr  = self.BFcr
        self.predict.BFmu  = self.BFmu
        self.predict.BFfat = self.BFfat
        self.predict.BFvsk = self.BFvsk

        self.predict.dQtot_dt= dQtot_dt/self.SA
        self.predict.Qtot = self.Qtot

        self.predict.SR    = self.SR
        self.predict.Esk   = self.Esk/self.SA
        self.predict.Emax  = self.Emax/self.SA
        self.predict.SkWet = self.SkWet*100.0
        self.predict.dMshiv= self.dMshiv/self.SA
        self.predict.MRC   = MRC/self.SA
        self.predict.CrSkdT= CrSkdT
        self.predict.RaSkdT= RaSkdT

        self.predict.FluidLoss= self.fluidLoss
        self.predict.O2debt   = self.O2debt
        self.predict.HR       = self.HR
        self.predict.SV       = self.SV
        self.predict.PSI      = self.PSI

        # setPts
        self.predict.ThypSet  = self.ThypSet
        self.predict.TimeOfDay= (self.time/60.0) + self.startTime

    ########################################################################
    # From here down, the compXXX and deriv methods are the same as before.
    ########################################################################

    # --------------- All your existing 'compKc', 'compKr', 'compStrokeVolume', ...
    # --------------- etc. remain the same. (Truncated for brevity, but
    # --------------- you should copy/paste them in full from your code.)
    def compKc(self, workMode, Vair, Vmove, Mrst):
        k1 = 1.96*(Vair**0.86)
        Kc = self.STANDARD_KC
        if workMode in ('r','a'):
            Kc = 11.6*math.sqrt(Vair)
        elif workMode == 't':
            Kc = 6.5*(Vmove**0.39) + k1
        elif workMode == 'f':
            Kc = 8.6*(Vmove**0.53) + k1
        elif workMode == 'e':
            Kc = 5.5 + k1
        elif workMode == 'm':
            if self.Mtot > Mrst:
                Kc = 8.6*(Vmove**0.53) + k1
            else:
                Kc = 11.6*math.sqrt(Vair)
        elif workMode == 'n':
            if self.Mtot > Mrst:
                Kc = 6.5*(Vmove**0.39) + k1
            else:
                Kc = 11.6*math.sqrt(Vair)
        return Kc

    def compKr(self, Kc, Iclo, Tmr, Ta):
        RadFact = 1.0 + 0.15*Iclo
        k1 = 4.0*Boltz*RadFact*SArad
        k2 = KClo*Iclo
        x1 = self.KR0
        ZERO = 1e-5
        reldif = 1.0
        ict = 0
        while reldif > ZERO:
            Kop = Kc + x1
            p = 1.0/(1.0 + k2*Kop)
            q = Tmr*x1 + Kc*Ta
            r = 1.0/Kop
            s = (1.0 - 0.5*p)*q*r + 0.5*self.Tsk*p + 273.0
            f = k1*(s**3) - x1
            dpdx = -k2/(1.0 + k2*Kop)**2
            dqdx = Tmr
            drdx = -1.0/(Kop*Kop)
            dsdx = (dqdx*r + drdx*q) - 0.5*(dpdx*q*r + p*dqdx*r + p*q*drdx) + 0.5*self.Tsk*dpdx
            dfdx = 3.0*k1*(s**2)*dsdx - 1.0
            x2 = x1 - f/dfdx
            reldif = abs(x2 - x1)
            x1 = x2
            ict+=1
            if ict == 6:
                raise ModelException(self.FAILED_TO_CONVERGE)
        return x1

    def compStrokeVolume(self, SVmax):
        SVlag = utils.LagIt(self.SVold, SVmax, float(self.time), 0.5)
        MWST  = self.Tsk
        if MWST > 38:
            MWST = 38
        if MWST <= 33:
            self.SV = SVlag
        else:
            self.SV = SVlag - 5.0*((SVmax - 85.0)/45.0)*(MWST-33.0)

    def compCOreq(self):
        COreq = self.BFcr + self.BFmuReq + self.BFfat + self.BFvskReq
        self.HR = COreq / self.SV
        self.BFra = COreq
        self.Overload = 0.0
        COmax = self.HRmax*self.SV
        if self.HR > self.HRmax:
            self.HR = self.HRmax
            self.BFra = COmax
            self.Overload = COreq - COmax
        self.PctHRmax = 100.0*self.HR/self.HRmax

    def compVascularBloodFlow(self, VO2):
        BFvskLast = self.BFvsk
        BFvskMax = utils.LagIt(self.BFvskMaxOld, self.BFvskMaxNew, float(self.time), 0.5)
        MWST = self.Tsk
        MWST = min(max(MWST, 30), 35.3)
        if VO2 <= 0.75:
            TsetBFnew = 37.07 - 0.108*(MWST-30)
        else:
            if MWST <= 33:
                TsetBFnew = 37.32 - 0.093*(MWST-30)
            else:
                TsetBFnew = 37.04 - 0.03*(MWST-33)
        TsetBF = utils.LagIt(self.TsetBFold, TsetBFnew, float(self.time), 1.0)
        PctBFvskMax = 70.3*(self.Tra - TsetBF)
        if PctBFvskMax>100:
            PctBFvskMax=100
        self.BFvskReq = max((PctBFvskMax/100.0)*BFvskMax, self.BFvskMin)
        if self.Overload > 0:
            self.BFvsk = BFvskLast
        else:
            self.BFvsk = self.BFvskReq

    def compVskSkCond(self, workMode):
        import math
        self.VskSkCond = self.SkCondConst * math.log(self.BFvsk/self.BFvskMin) + (10.0*self.SA)
        if workMode == 'r':
            PctBFcrRst = 100.0 - 1.266*(self.PctHRmax-25.0)
        else:
            PctBFcrRst = 100.0 - 1.086*(self.PctHRmax-39.0)
        if PctBFcrRst>100:
            PctBFcrRst=100
        BFcrNew = self.COrst*(fCOrstCr)*(PctBFcrRst/100.0)
        self.BFcr = utils.LagIt(self.BFcrOld, BFcrNew, float(self.time), 1.0)

    def compMuscleBloodFlow(self):
        self.BFmuReq = (self.EEmu*self.O2Equiv)/(0.01*self.PctExt*self.BlO2Cap)
        self.BFmu = self.BFmuReq
        MUmin = (self.EEmu*self.O2Equiv)/(self.BlO2Cap)
        MUpot = self.BFmuReq - MUmin
        self.PctExt = 80
        self.LeftOver=0.0
        if self.Overload>0 and MUpot>0:
            if self.Overload<=MUpot:
                self.BFmu = self.BFmuReq - self.Overload
                self.PctExt = (100.0*self.EEmu*self.O2Equiv)/(self.BFmu*self.BlO2Cap)
            else:
                self.LeftOver = self.Overload - MUpot
                self.BFmu = MUmin - self.LeftOver
                self.PctExt = 100.0
        elif self.Overload>0 and MUpot<=0:
            self.LeftOver = self.Overload
            self.BFmu     = MUmin - self.LeftOver
            self.PctExt   = 100.0

    def compShivering(self):
        SkSig = self.Tsk - self.TskSet
        CldSkSig = -SkSig if SkSig<=0 else 0.0
        HypSig = self.Tra - self.ThypSet
        CldHypSig = -HypSig if HypSig<=0 else 0.0
        self.dMshiv = (self.SA*19.4*CldSkSig*CldHypSig)

    def compSweatRate(self):
        SkSig = self.Tsk - self.TskSet
        HypSig= self.Tra - self.ThypSet
        if SkSig>100:
            self.SR = self.SRmax
        else:
            pctWgtLoss = 0.1*self.fluidLoss/self.BW
            deltaAlphaSR = self.DELTA_ALPHA_SR*pctWgtLoss
            deltaHypSig  = self.DELTA_HYPSIG*pctWgtLoss
            self.SR = self.SA*((4.83+deltaAlphaSR)*(HypSig+deltaHypSig)
                    + 0.56*SkSig)*math.exp(SkSig/10.0)
        self.SR = max(min(self.SR, self.SRmax), self.SRmin)

    def compDripSkWetEsk(self, Kop, Kc, Pvap, ClEffFact, Im):
        self.Drip=0.0
        self.SkWet=0.0
        Esw = HtVap*self.SR
        ClPermFact = (Kop/Kc)*ClEffFact*Im
        Psk = utils.SatVP(self.Tsk)
        self.Emax = self.SA*Lewis*Kc*(Psk-Pvap)*ClPermFact
        if self.Emax<=0:
            self.Emax=0
            Esw=0
            self.SkWet=1
            self.Drip=self.SR
        else:
            self.SkWet=Esw/self.Emax
            if self.SkWet>1:
                Esw=self.Emax
                self.SkWet=1
                self.Drip=self.SR-(self.Emax/HtVap)
            else:
                self.Drip=0
        self.Esk=Esw

    def compShiverCorrection(self, Mmu2, Mtot2, Mext, EEmuOld, EEmuNew):
        self.Mmu  = Mmu2 + self.dMshiv
        self.Mtot = Mtot2+ self.dMshiv
        self.Mnet = self.Mtot - Mext
        Tlag = self.time - self.T_EEmuNew
        EEmu2 = utils.LagIt(EEmuOld, EEmuNew, float(Tlag), 1.0)
        self.EEmu = EEmu2 + self.dMshiv

    def compDryHeatLoss(self, RadFact, Kc, Tmr, Ta, Iclo):
        self.Tcl = self.Top + (self.ClEffFact*(self.Tsk - self.Top))
        Kr = 4.0*Boltz*(((self.Tcl+self.Top)/2.0+273.0)**3)*RadFact*SArad
        Kop = Kc+Kr
        self.Top = (Kr*Tmr + Kc*Ta)/Kop
        self.ClEffFact = 1.0/(1.0 + Kop*KClo*Iclo)
        self.DryHtLoss1 = self.SA*Kop*self.ClEffFact*(self.Tsk - self.Top)
        self.DryHtLoss2 = self.SA*Kop*self.ClEffFact*(self.TTC - self.Top)

    def deriv(self, Cres, Eres, Mcr, Mfat, Mvsk, Msk):
        self.dQra_dt = SpHtBl*0.001*(self.BFcr*(self.Tcr-self.Tra)
                      + self.BFmu*(self.Tmu-self.Tra)
                      + self.BFfat*(self.Tfat-self.Tra)
                      + self.BFvsk*(self.Tvsk-self.Tra)) - (Cres+Eres)

        self.dQcr_dt = Mcr - self.CrMuCond*(self.Tcr-self.Tmu) \
                       - SpHtBl*0.001*self.BFcr*(self.Tcr-self.Tra)
        self.dQmu_dt = self.Mmu + self.CrMuCond*(self.Tcr-self.Tmu) \
                       - self.MuFatCond*(self.Tmu-self.Tfat) \
                       - SpHtBl*0.001*self.BFmu*(self.Tmu-self.Tra)
        self.dQfat_dt= Mfat + self.MuFatCond*(self.Tmu-self.Tfat) \
                       - self.FatVskCond*(self.Tfat-self.Tvsk) \
                       - SpHtBl*0.001*self.BFfat*(self.Tfat-self.Tra)
        self.dQvsk_dt= Mvsk + self.FatVskCond*(self.Tfat-self.Tvsk) \
                       - self.VskSkCond*(self.Tvsk-self.Tsk) \
                       - SpHtBl*0.001*self.BFvsk*(self.Tvsk-self.Tra)
        self.dQsk_dt = Msk + self.VskSkCond*(self.Tvsk-self.Tsk) \
                       - self.DryHtLoss1 - self.Esk
        self.dQTC_dt = Msk + self.VskSkCond*(self.Tvsk-self.TTC) \
                       - self.DryHtLoss2

        self.dO2debt_dt = 0.0001*self.PctExt*self.BlO2Cap*self.LeftOver

    def compTbdy(self):
        totHtCap = (self.HtCapRA + self.HtCapCr + self.HtCapMu
                    + self.HtCapFat + self.HtCapVsk + self.HtCapSk)
        self.Tbdy = (
            self.HtCapRA*self.Tra + self.HtCapCr*self.Tcr
            + self.HtCapMu*self.Tmu + self.HtCapFat*self.Tfat
            + self.HtCapVsk*self.Tvsk + self.HtCapSk*self.Tsk
        )/totHtCap

    def log_state(self):
        self.states["time"].append(self.time)
        self.states["Tra"].append(self.Tra)
        self.states["Tcr"].append(self.Tcr)
        self.states["Tmu"].append(self.Tmu)
        self.states["Tfat"].append(self.Tfat)
        self.states["Tvsk"].append(self.Tvsk)
        self.states["Tsk"].append(self.Tsk)
        self.states["TTC"].append(self.TTC)
        self.states["BFra"].append(self.BFra)
        self.states["BFcr"].append(self.BFcr)
        self.states["BFmu"].append(self.BFmu)
        self.states["BFfat"].append(self.BFfat)
        self.states["BFvsk"].append(self.BFvsk)
        self.states["HR"].append(self.HR)
        self.states["SV"].append(self.SV)
        CO = self.HR*self.SV
        self.states["CO"].append(CO)
        CI = CO/self.SA if self.SA>0 else 0.0
        self.states["CI"].append(CI)
        self.states["SR"].append(self.SR)
        self.states["Esk"].append(self.Esk)
        self.states["Emax"].append(self.Emax)
        self.states["SkWet"].append(self.SkWet)
        self.states["dMshiv"].append(self.dMshiv)
        self.states["fluidLoss"].append(self.fluidLoss)
        self.states["O2debt"].append(self.O2debt)
        self.states["PSI"].append(self.PSI)
        self.states["Qra"].append(self.Qra)
        self.states["Qcr"].append(self.Qcr)
        self.states["Qmu"].append(self.Qmu)
        self.states["Qfat"].append(self.Qfat)
        self.states["Qvsk"].append(self.Qvsk)
        self.states["Qsk"].append(self.Qsk)
        self.states["Qtot"].append(self.Qtot)
        self.states["Tbdy"].append(self.Tbdy)
        self.states["Overload"].append(self.Overload)
        self.states["PctExt"].append(self.PctExt)
        self.states["dQra_dt"].append(self.dQra_dt)
        self.states["dQcr_dt"].append(self.dQcr_dt)
        self.states["dQmu_dt"].append(self.dQmu_dt)
        self.states["dQfat_dt"].append(self.dQfat_dt)
        self.states["dQvsk_dt"].append(self.dQvsk_dt)
        self.states["dQsk_dt"].append(self.dQsk_dt)
        self.states["dQTC_dt"].append(self.dQTC_dt)

def main():
    """
    Minimal demonstration with visualization
    (One 60-min segment, fairly hot environment).
    """

    segments = {
        'time'          : [1] * 60,
        "BW"            : [77.61] * 60,
        "SA"            : [1.8] * 60,
        "AGE"           : [44] * 60,
        "Ta"            : [30.0] * 60,
        "Tmr"           : [35.0] * 60,
        "Vair"          : [0.1] * 60,
        "Pvap"          : [25.0] * 60,
        "Iclo"          : [0.5] * 60,
        "Im"            : [0.5] * 60,
        "PctFat"        : [20.0] * 60,
        "Mtot"          : [400.0] * 60,
        "Mrst"          : [100.0] * 60,
        "Mext"          : [0] * 60,
        "Vmove"         : [1.0] * 60,
        "workMode"      : ['f'] * 60,
        "fluidIntake"   : [0.0] * 60,
        "acclimIndex"   : [NO_ACCLIMATION] * 60,
        "dehydIndex"    : [DEHYD_NORMAL] * 60,
        "startTime"     : [0.0] * 60,
        "Tcr0"          : [37.06] * 60,
        "circadianModel": [False] * 60,
        "tcoreOverride" : [True] * 60
    }

    model = ScenarioModel(segments)

    # Extract time series from model.states
    time      = model.states["time"]
    Tcr       = model.states["Tcr"]
    HR        = model.states["HR"]
    FluidLoss = model.states["fluidLoss"]
    Tsk       = model.states["Tsk"]

    print("---------------", len(time))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(time, Tcr, label='Core Temp (Tcr)')
    axs[0, 0].set_xlabel('Time (min)')
    axs[0, 0].set_ylabel('Temp (C)')
    axs[0, 0].set_title('Core Temperature Over Time')
    axs[0, 0].legend()

    axs[0, 1].plot(time, HR, label='Heart Rate', color='orange')
    axs[0, 1].set_xlabel('Time (min)')
    axs[0, 1].set_ylabel('bpm')
    axs[0, 1].set_title('Heart Rate Over Time')
    axs[0, 1].legend()

    axs[1, 0].plot(time, FluidLoss, label='Fluid Loss', color='green')
    axs[1, 0].set_xlabel('Time (min)')
    axs[1, 0].set_ylabel('Fluid Loss (g)')
    axs[1, 0].set_title('Fluid Loss Over Time')
    axs[1, 0].legend()

    axs[1, 1].plot(time, Tsk, label='Skin Temp (Tsk)', color='red')
    axs[1, 1].set_xlabel('Time (min)')
    axs[1, 1].set_ylabel('Temp (C)')
    axs[1, 1].set_title('Skin Temperature Over Time')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Optionally save figure
    fig.savefig('scenario_model_output.png')

    # Final results
    p = model.getPredict()
    print("=== Final Predictions ===")
    print(f"  Tcr = {p.Tcr:.2f} C")
    print(f"  Tsk = {p.Tsk:.2f} C")
    print(f"  HR  = {p.HR:.2f} bpm")
    print(f"  PSI = {p.PSI:.2f}")
    print(f"  FluidLoss= {p.FluidLoss:.2f} g")

if __name__ == "__main__":
    main()
