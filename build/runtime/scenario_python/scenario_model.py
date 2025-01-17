"""
scenario_model.py

A Python module that implements the logic from ScenarioModel.java as closely
as possible, using the Python 'scenario_constants', 'tissue_conductance', and 'utils'
modules. 

Note: This is quite a large code translation from Java, with a 2-step integration
scheme in step(). 
"""

import math

from scenario_constants import (
    # Constants:
    # scenario_constants includes many from ScenarioConstants.java
    NO_ACCLIMATION, PART_ACCLIMATION, FULL_ACCLIMATION,
    DEHYD_NORMAL, DEHYD_MODERATE, DEHYD_SEVERE,
    fMrstCr, fMrstMu, fMrstFat, fMrstVsk, fMrstSk,
    fCOrstCr, fCOrstMu, fCOrstFat, fCOrstVsk,
    SpHtCr, SpHtMu, SpHtFat, SpHtSk, SpHtBl,
    Lewis, Boltz, SArad, KClo, HtVap
)
from tissue_conductance import TissueConductance
import utils  # for SatVP, LagIt, etc.
import matplotlib.pyplot as plt

# We'll define "Predict" as a simple container for final outputs
class Predict:
    def __init__(self):
        # a subset of final "time dependent" or "time independent" variables
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

        # We also store setpoints if desired
        self.ThypSet  = 0.0
        self.TskSet   = 0.0
        self.TcrFlag  = 0.0
        self.HRFlag   = 0.0
        self.dQ_dtFlag= 0.0
        self.QtotFlag = 0.0
        self.TimeOfDay= 0.0

# Minimally define input containers
class Inputs1:
    """Mirrors the Java Inputs1 class, storing environment and subject data."""
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
    """Mirrors Java Inputs2, storing the metabolic/work data."""
    def __init__(self, Mtot=200.0, Mrst=100.0, Mext=20.0, Vmove=1.0,
                 workMode='r', fluidIntake=0.0):
        self.Mtot        = Mtot
        self.Mrst        = Mrst
        self.Mext        = Mext
        self.Vmove       = Vmove
        self.workMode    = workMode  # e.g. 'r', 't', 'f', etc.
        self.fluidIntake = fluidIntake

class Inputs3:
    """Mirrors Java Inputs3, storing acclimation, dehydration, circadian, etc."""
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
    """A custom exception to mirror the Java ScenarioModel's ModelException."""
    pass

class ScenarioModel:

    # Additional Java constants that aren't in scenario_constants.py
    # for dryness:
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

    """Translation of ScenarioModel.java to Python."""
    def __init__(self):
        self.predict = Predict()

        # We'll store states in a dictionary
        self.states = {}
        # We'll define each key as an empty list. 
        # You can define them now or lazily. Here we define them upfront:
        self.states["time"]   = []
        self.states["Tra"]    = []
        self.states["Tcr"]    = []
        self.states["Tmu"]    = []
        self.states["Tfat"]   = []
        self.states["Tvsk"]   = []
        self.states["Tsk"]    = []
        self.states["TTC"]    = []
        self.states["BFra"]   = []
        self.states["BFcr"]   = []
        self.states["BFmu"]   = []
        self.states["BFfat"]  = []
        self.states["BFvsk"]  = []
        self.states["HR"]     = []
        self.states["SV"]     = []
        self.states["CO"]     = []
        self.states["CI"]     = []  # Cardiac index = CO / SA
        self.states["SR"]     = []
        self.states["Esk"]    = []
        self.states["Emax"]   = []
        self.states["SkWet"]  = []
        self.states["dMshiv"] = []
        self.states["fluidLoss"] = []
        self.states["O2debt"] = []
        self.states["PSI"]    = []
        self.states["Qra"]    = []
        self.states["Qcr"]    = []
        self.states["Qmu"]    = []
        self.states["Qfat"]   = []
        self.states["Qvsk"]   = []
        self.states["Qsk"]    = []
        self.states["Qtot"]   = []
        self.states["Tbdy"]   = []
        # add more if you want, e.g. conduction or Overload, PctExt, etc.
        self.states["Overload"]   = []
        self.states["PctExt"]     = []
        self.states["dQra_dt"]    = []
        self.states["dQcr_dt"]    = []
        self.states["dQmu_dt"]    = []
        self.states["dQfat_dt"]   = []
        self.states["dQvsk_dt"]   = []
        self.states["dQsk_dt"]    = []
        self.states["dQTC_dt"]    = []

        # time independent
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

        # time dependent
        self.Tra  = 0.0
        self.Tcr  = 0.0
        self.Tmu  = 0.0
        self.Tfat = 0.0
        self.Tvsk = 0.0
        self.Tsk  = 0.0
        self.Tcl  = 0.0

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
        self.TTC   = 0.0
        self.BFvskReq= 0.0
        self.BFmuReq = 0.0
        self.BFvskMaxNew = 0.0
        self.EEmu  = 0.0
        self.SV    = 0.0
        self.HR    = 0.0
        self.PSI   = 0.0
        self.Tbdy  = 0.0

        # new class data
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

        # derivatives
        self.dQra_dt   = 0.0
        self.dQcr_dt   = 0.0
        self.dQmu_dt   = 0.0
        self.dQfat_dt  = 0.0
        self.dQvsk_dt  = 0.0
        self.dQsk_dt   = 0.0
        self.dQTC_dt   = 0.0
        self.dO2debt_dt= 0.0

        # circadian
        self.delThypSet   = 0.0
        self.acclimIndex  = NO_ACCLIMATION
        self.dehydIndex   = DEHYD_NORMAL
        self.startTime    = 0.0
        self.Tcr0         = 37.0
        self.circadianModel = False
        self.tcoreOverride  = False
        self.Tre0  = 0.0
        self.HR0   = 0.0

        # loop control
        self.dt   = 0.025
        self.FUZZ = 0.00025
        self.time = 0.0
        self.T_EEmuNew = 0.0
        self.iter = 0
        self.DEBUG = False
        self.stabilityFactor = 0.5

    def __init__(self, segments):
        self.predict = Predict()

        # We'll store states in a dictionary
        self.states = {}
        # We'll define each key as an empty list. 
        # You can define them now or lazily. Here we define them upfront:
        self.states["time"]   = []
        self.states["Tra"]    = []
        self.states["Tcr"]    = []
        self.states["Tmu"]    = []
        self.states["Tfat"]   = []
        self.states["Tvsk"]   = []
        self.states["Tsk"]    = []
        self.states["TTC"]    = []
        self.states["BFra"]   = []
        self.states["BFcr"]   = []
        self.states["BFmu"]   = []
        self.states["BFfat"]  = []
        self.states["BFvsk"]  = []
        self.states["HR"]     = []
        self.states["SV"]     = []
        self.states["CO"]     = []
        self.states["CI"]     = []  # Cardiac index = CO / SA
        self.states["SR"]     = []
        self.states["Esk"]    = []
        self.states["Emax"]   = []
        self.states["SkWet"]  = []
        self.states["dMshiv"] = []
        self.states["fluidLoss"] = []
        self.states["O2debt"] = []
        self.states["PSI"]    = []
        self.states["Qra"]    = []
        self.states["Qcr"]    = []
        self.states["Qmu"]    = []
        self.states["Qfat"]   = []
        self.states["Qvsk"]   = []
        self.states["Qsk"]    = []
        self.states["Qtot"]   = []
        self.states["Tbdy"]   = []
        # add more if you want, e.g. conduction or Overload, PctExt, etc.
        self.states["Overload"]   = []
        self.states["PctExt"]     = []
        self.states["dQra_dt"]    = []
        self.states["dQcr_dt"]    = []
        self.states["dQmu_dt"]    = []
        self.states["dQfat_dt"]   = []
        self.states["dQvsk_dt"]   = []
        self.states["dQsk_dt"]    = []
        self.states["dQTC_dt"]    = []

        # time independent
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

        # time dependent
        self.Tra  = 0.0
        self.Tcr  = 0.0
        self.Tmu  = 0.0
        self.Tfat = 0.0
        self.Tvsk = 0.0
        self.Tsk  = 0.0
        self.Tcl  = 0.0

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
        self.TTC   = 0.0
        self.BFvskReq= 0.0
        self.BFmuReq = 0.0
        self.BFvskMaxNew = 0.0
        self.EEmu  = 0.0
        self.SV    = 0.0
        self.HR    = 0.0
        self.PSI   = 0.0
        self.Tbdy  = 0.0

        # new class data
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

        # derivatives
        self.dQra_dt   = 0.0
        self.dQcr_dt   = 0.0
        self.dQmu_dt   = 0.0
        self.dQfat_dt  = 0.0
        self.dQvsk_dt  = 0.0
        self.dQsk_dt   = 0.0
        self.dQTC_dt   = 0.0
        self.dO2debt_dt= 0.0

        # circadian
        self.delThypSet   = 0.0
        self.acclimIndex  = NO_ACCLIMATION
        self.dehydIndex   = DEHYD_NORMAL
        self.startTime    = 0.0
        self.Tcr0         = 37.0
        self.circadianModel = False
        self.tcoreOverride  = False
        self.Tre0  = 0.0
        self.HR0   = 0.0

        # loop control
        self.dt   = 0.025
        self.FUZZ = 0.00025
        self.time = 0.0
        self.T_EEmuNew = 0.0
        self.iter = 0
        self.DEBUG = False
        self.stabilityFactor = 0.5

        # Initialize the model with the segments
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

        i1, i2, i3 = get_inputs_from_segments(segments, 0)
        self.init(i1, i2, i3)
        T_next = 0
        for i in range(0, len(segments['time'])):
            i1, i2, i3 = get_inputs_from_segments(segments, i)
            T_next = T_next + segments['time'][i]
            self.step(i1, i2, T_next)

    def getPredict(self):
        return self.predict

    def init(self, inputs1, inputs2, inputs3):
        """
        Translated from ScenarioModel.java init(...).
        """
        # Extract Inputs1
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

        # Extract Inputs2
        self.Mtot = inputs2.Mtot
        Mrst      = inputs2.Mrst
        Mext      = inputs2.Mext
        self.Mnet = self.Mtot - Mext
        Vmove     = inputs2.Vmove
        workMode  = inputs2.workMode
        fluidIntake = inputs2.fluidIntake

        # Extract Inputs3
        self.acclimIndex  = inputs3.acclimIndex
        self.dehydIndex   = inputs3.dehydIndex
        self.startTime    = inputs3.startTime
        self.Tcr0         = inputs3.Tcr0
        self.circadianModel = inputs3.circadianModel
        self.tcoreOverride  = inputs3.tcoreOverride

        # circadian stub
        # If you had a real circadian table, you'd load it. 
        # We'll just do a trivial approach:
        def circadianInterpTime(t):
            # just a dummy—always returns 36.96
            return 36.96

        self.ThypSet = self.STANDARD_THYP_SET
        if self.circadianModel:
            self.ThypSet = circadianInterpTime(self.startTime)

        # Setup initial node temps based on acclim
        if self.acclimIndex == NO_ACCLIMATION:
            self.delThypSet = 0.0
            self.Tra  = 36.75
            self.Tmu  = 36.07
            self.Tcr  = 36.98
            self.Tfat = 33.92
            self.Tvsk = 33.49
            self.Tsk  = 33.12
            self.TTC  = self.Tsk
        elif self.acclimIndex == PART_ACCLIMATION:
            self.delThypSet = -0.25
            self.Tra  = 36.73
            self.Tmu  = 36.10
            self.Tcr  = 36.96
            self.Tfat = 34.13
            self.Tvsk = 33.73
            self.Tsk  = 32.87
            self.TTC  = self.Tsk
        elif self.acclimIndex == FULL_ACCLIMATION:
            self.delThypSet = -0.5
            self.Tra  = 36.49
            self.Tmu  = 35.91
            self.Tcr  = 36.72
            self.Tfat = 34.04
            self.Tvsk = 33.66
            self.Tsk  = 32.79
            self.TTC  = self.Tsk

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

        # blood flow distribution
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

        # heat capacities
        self.HtCapRA = SpHtBl * 0.02 * Wnfat / 1000.0  # from Java code: fBWnfatRa * Wnfat
        self.HtCapCr = SpHtCr * 0.37 * Wnfat / 1000.0
        self.HtCapMu = SpHtMu * 0.54 * Wnfat / 1000.0
        self.HtCapFat= SpHtFat* Wfat / 1000.0
        self.HtCapVsk= SpHtBl * 0.01 * Wnfat / 1000.0
        self.HtCapSk = SpHtSk * 0.06 * Wnfat / 1000.0

        # compute Kc, Kr
        Kc = self.compKc(workMode, Vair, Vmove, Mrst)
        Kr = self.compKr(Kc, Iclo, Tmr, Ta)
        Kop = Kr + Kc
        self.Top = (Kr * Tmr + Kc * Ta) / Kop
        self.ClEffFact = 1.0 / (1.0 + Kop * KClo * Iclo)
        self.Tcl = self.Top + (self.ClEffFact*(self.Tsk - self.Top))

        # Eres, Cres
        Eres = self.SA * 0.0023 * self.Mnet * (44.0 - Pvap)
        Cres = self.SA * 0.0012 * self.Mnet * (34.0 - Ta)

        self.DryHtLoss1 = self.SA * Kop * self.ClEffFact * (self.Tsk - self.Top)
        self.DryHtLoss2 = self.SA * Kop * self.ClEffFact * (self.TTC - self.Top)
        MRC = self.Mnet - (Eres + Cres) - self.DryHtLoss1
        ClPermFact = (Kop / Kc)* self.ClEffFact* Im
        Psk = utils.SatVP(self.Tsk)
        self.Emax = self.SA*Lewis*Kc*(Psk - Pvap)*ClPermFact

        # initial 
        self.SR = 0.0
        self.Esk= 0.0
        self.dMshiv= 0.0
        self.PctExt = 80.0
        self.LeftOver= 0.0
        self.time = 0.0
        self.T_EEmuNew= 0.0
        self.EEmu = self.MrstMu + Mext

        EEmuOld = self.EEmu
        # compShiverCorrection once
        self.compShiverCorrection(self.MrstMu, self.Mtot, Mext, EEmuOld, EEmuOld)
        # do derivative
        self.deriv(Cres, Eres, self.MrstCr, self.MrstFat, self.MrstVsk, self.MrstSk)

        # Just store initial results in self.predict
        # (like Java does with Predict1..7)
        # We'll mirror it in a single container:

        # We'll zero out Q's
        self.Qra=0.0
        self.Qcr=0.0
        self.Qmu=0.0
        self.Qfat=0.0
        self.Qvsk=0.0
        self.Qsk=0.0
        self.Qtot=0.0

        # SV stuff
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

        # set flags
        self.HRmax = 220.0 - age
        # Done with init; set dt, fuzz
        self.iter = 0
        self.dt   = 0.025
        self.FUZZ = 0.01*self.dt

    def compKc(self, workMode, Vair, Vmove, Mrst):
        """
        Java: compKc(char workMode, float Vair, float Vmove, float Mrst)
        We replicate the switch-case logic from ScenarioModel.java
        """
        k1 = 1.96*(Vair**0.86)
        Kc = self.STANDARD_KC  # default
        # parse workMode
        if workMode in ('r','a'):
            # resting or something
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
        """
        Java: compKr(...) uses Newton's method to solve for Kr.
        We'll replicate that approach.
        """
        RadFact = 1.0 + 0.15*Iclo
        k1 = 4.0*Boltz*RadFact*SArad
        k2 = KClo*Iclo
        # initial guess
        x1 = self.KR0
        if self.time > 120.0:
            x1 = self.KR0
        # The iterative approach
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
            # derivatives
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
        """
        Java: compStrokeVolume(float SVmax)
        uses LagIt on SVold -> SVmax
        """
        SVlag = utils.LagIt(self.SVold, SVmax, float(self.time), 0.5)
        MWST  = self.Tsk
        if MWST > 38: 
            MWST = 38
        if MWST <= 33:
            self.SV = SVlag
        else:
            self.SV = SVlag - 5.0*((SVmax - 85.0)/45.0)*(MWST-33.0)

    def compCOreq(self):
        """
        Java: compCOreq()
        """
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
        """
        Java: compVascularBloodFlow(float VO2)
        modifies BFvsk, BFvskReq
        """
        BFvskLast = self.BFvsk
        BFvskMax = utils.LagIt(self.BFvskMaxOld, self.BFvskMaxNew, float(self.time), 0.5)

        MWST = self.Tsk
        if MWST < 30:
            MWST = 30
        if MWST > 35.3:
            MWST = 35.3

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
        self.BFvskReq = (PctBFvskMax/100.0)*BFvskMax
        if self.BFvskReq < self.BFvskMin:
            self.BFvskReq = self.BFvskMin

        if self.Overload > 0:
            self.BFvsk = BFvskLast
        else:
            self.BFvsk = self.BFvskReq

    def compVskSkCond(self, workMode):
        """
        Java: compVskSkCond(char workMode)
        modifies VskSkCond, BFcr, BFcrOld
        """
        import math
        self.VskSkCond = self.SkCondConst * math.log(self.BFvsk/self.BFvskMin) + (10.0*self.SA)
        # core BF
        if workMode == 'r':
            # resting or sitting in heat
            PctBFcrRst = 100.0 - 1.266*(self.PctHRmax-25.0)
        else:
            # exercise
            PctBFcrRst = 100.0 - 1.086*(self.PctHRmax-39.0)
        if PctBFcrRst>100:
            PctBFcrRst=100
        BFcrNew = self.COrst*(fCOrstCr)*(PctBFcrRst/100.0)
        self.BFcr = utils.LagIt(self.BFcrOld, BFcrNew, float(self.time), 1.0)

    def compMuscleBloodFlow(self):
        """
        Java: compMuscleBloodFlow()
        modifies BFmu, BFmuReq, PctExt, LeftOver
        """
        self.BFmuReq = (self.EEmu*self.O2Equiv)/(0.01*self.PctExt*self.BlO2Cap)
        self.BFmu = self.BFmuReq
        MUmin = (self.EEmu*self.O2Equiv)/(self.BlO2Cap)  # 100% extraction
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
        """
        Java: compShivering() modifies dMshiv
        """
        SkSig = self.Tsk - self.TskSet
        if SkSig<=0:
            CldSkSig = -SkSig
        else:
            CldSkSig = 0.0

        HypSig = self.Tra - self.ThypSet
        if HypSig<=0:
            CldHypSig = -HypSig
        else:
            CldHypSig = 0.0

        self.dMshiv = (self.SA*19.4*CldSkSig*CldHypSig)

    def compSweatRate(self):
        """
        Java: compSweatRate()
        modifies SR
        """
        SkSig = self.Tsk - self.TskSet
        HypSig= self.Tra - self.ThypSet
        if SkSig>100:
            self.SR = self.SRmax
        else:
            pctWgtLoss = 0.1*self.fluidLoss/self.BW
            deltaAlphaSR = self.DELTA_ALPHA_SR*pctWgtLoss
            deltaHypSig  = self.DELTA_HYPSIG   *pctWgtLoss
            self.SR = self.SA*((4.83+deltaAlphaSR)*(HypSig+deltaHypSig)
                    + 0.56*SkSig)*math.exp(SkSig/10.0)
        if self.SR<self.SRmin:
            self.SR=self.SRmin
        if self.SR>self.SRmax:
            self.SR=self.SRmax

    def compDripSkWetEsk(self, Kop, Kc, Pvap, ClEffFact, Im):
        """
        Java: compDripSkWetEsk(...)
        modifies Drip, SkWet, Esk, Emax
        """
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
        # Edif = 0.06*(1.0-self.SkWet)*self.Emax  # legacy?
        self.Esk=Esw

    def compShiverCorrection(self, Mmu2, Mtot2, Mext, EEmuOld, EEmuNew):
        """
        Java: compShiverCorrection(...)
        modifies Mmu, Mtot, Mnet, EEmu
        """
        self.Mmu  = Mmu2 + self.dMshiv
        self.Mtot = Mtot2+ self.dMshiv
        self.Mnet = self.Mtot - Mext
        Tlag = self.time - self.T_EEmuNew
        EEmu2 = utils.LagIt(EEmuOld, EEmuNew, float(Tlag), 1.0)
        self.EEmu = EEmu2 + self.dMshiv

    def compDryHeatLoss(self, RadFact, Kc, Tmr, Ta, Iclo):
        """
        Java: compDryHeatLoss(...)
        modifies Tcl, DryHtLoss1, DryHtLoss2, Top, ClEffFact
        """
        self.Tcl = self.Top + (self.ClEffFact*(self.Tsk - self.Top))
        Kr = 4.0*Boltz*(((self.Tcl+self.Top)/2.0+273.0)**3)*RadFact*SArad
        Kop = Kc+Kr
        self.Top = (Kr*Tmr + Kc*Ta)/Kop
        self.ClEffFact = 1.0/(1.0 + Kop*KClo*Iclo)
        self.DryHtLoss1 = self.SA*Kop*self.ClEffFact*(self.Tsk - self.Top)
        self.DryHtLoss2 = self.SA*Kop*self.ClEffFact*(self.TTC - self.Top)

    def deriv(self, Cres, Eres, Mcr, Mfat, Mvsk, Msk):
        """
        Java: deriv(...)
        modifies dQra_dt, dQcr_dt, dQmu_dt, dQfat_dt, dQvsk_dt, dQsk_dt, dQTC_dt
        """
        # from ScenarioModel.java
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
        """
        Java: compTbdy()
        modifies Tbdy
        """
        totHtCap = (self.HtCapRA + self.HtCapCr + self.HtCapMu
                    + self.HtCapFat + self.HtCapVsk + self.HtCapSk)
        self.Tbdy = (self.HtCapRA*self.Tra + self.HtCapCr*self.Tcr
                   + self.HtCapMu*self.Tmu + self.HtCapFat*self.Tfat
                   + self.HtCapVsk*self.Tvsk + self.HtCapSk*self.Tsk)/totHtCap

    def compTmin0(self):
        """
        Translated from ScenarioModel.java's compTmin0().
        Computes static time constants to get an initial minimal time step.
        """
        # Example logic:
        # core node
        Tmu_cr = self.HtCapCr / (self.CrMuCond if self.CrMuCond != 0 else 9999.0)
        Tmin = Tmu_cr

        # muscle node
        Tcr_mu = self.HtCapMu / (self.CrMuCond if self.CrMuCond != 0 else 9999.0)
        if Tcr_mu < Tmin:
            Tmin = Tcr_mu
        Tfat_mu = self.HtCapMu / (self.MuFatCond if self.MuFatCond != 0 else 9999.0)
        if Tfat_mu < Tmin:
            Tmin = Tfat_mu

        # fat node
        Tmu_fat = self.HtCapFat / (self.MuFatCond if self.MuFatCond != 0 else 9999.0)
        if Tmu_fat < Tmin:
            Tmin = Tmu_fat
        Tvsk_fat = self.HtCapFat / (self.FatVskCond if self.FatVskCond != 0 else 9999.0)
        if Tvsk_fat < Tmin:
            Tmin = Tvsk_fat

        # Rbl_fat
        if self.BFfat > 0:
            Rbl_fat = 1.0 / (SpHtBl * 0.001 * self.BFfat)
        else:
            Rbl_fat = 9999.0
        Tbl_fat = Rbl_fat * self.HtCapFat
        if Tbl_fat < Tmin:
            Tmin = Tbl_fat

        # vascular skin node
        Tfat_vsk = self.HtCapVsk / (self.FatVskCond if self.FatVskCond != 0 else 9999.0)
        if Tfat_vsk < Tmin:
            Tmin = Tfat_vsk

        # blood node
        Tfat_bl = Rbl_fat * self.HtCapRA
        if Tfat_bl < Tmin:
            Tmin = Tfat_bl

        return Tmin
    
    def compTmin(self):
        """
        Translated from ScenarioModel.java's compTmin().

        This function computes additional 'dynamic' time constants
        and returns the overall minimum. In Java, it also can write
        data to 'tau.dat' if PRINT_TAU is true. Here in Python, we omit
        the file writing by default, but keep the logic intact.
        """
        # In Java, they start with "Tmin = Tmin0;". We'll do the same:
        Tmin = self.compTmin0()

        # We won't do actual file output, but if you want to replicate
        # the printing in Java, you can do so here.

        # The code uses local variables like Rbl_cr, Rbl_mu, Rbl_vsk, etc.:
        # Rbl_xxx = 1 / (SpHtBl * 0.001f * BFxxx)

        # 1) core node
        if self.BFcr > 0:
            Rbl_cr = 1.0 / (SpHtBl * 0.001 * self.BFcr)
        else:
            Rbl_cr = 9999.0  # in case BFcr == 0

        Tbl = Rbl_cr * self.HtCapCr
        if Tbl < Tmin:
            Tmin = Tbl

        # 2) muscle node
        if self.BFmu > 0:
            Rbl_mu = 1.0 / (SpHtBl * 0.001 * self.BFmu)
        else:
            Rbl_mu = 9999.0

        Tbl = Rbl_mu * self.HtCapMu
        if Tbl < Tmin:
            Tmin = Tbl

        # 3) vascular skin node
        if self.BFvsk > 0:
            Rbl_vsk = 1.0 / (SpHtBl * 0.001 * self.BFvsk)
        else:
            Rbl_vsk = 9999.0

        Tbl = Rbl_vsk * self.HtCapVsk
        if Tbl < Tmin:
            Tmin = Tbl

        # Tsk_vsk = HtCapVsk / VskSkCond
        if self.VskSkCond != 0:
            Tsk_vsk = self.HtCapVsk / self.VskSkCond
            if Tsk_vsk < Tmin:
                Tmin = Tsk_vsk

        # 4) skin node
        # Tvsk_sk = HtCapSk / VskSkCond
        if self.VskSkCond != 0:
            Tvsk_sk = self.HtCapSk / self.VskSkCond
            if Tvsk_sk < Tmin:
                Tmin = Tvsk_sk

        # 5) blood compartment
        #   a) Rbl_cr * HtCapRA
        Tbl = Rbl_cr * self.HtCapRA
        if Tbl < Tmin:
            Tmin = Tbl

        #   b) Rbl_mu * HtCapRA
        Tbl = Rbl_mu * self.HtCapRA
        if Tbl < Tmin:
            Tmin = Tbl

        #   c) Rbl_vsk * HtCapRA
        Tbl = Rbl_vsk * self.HtCapRA
        if Tbl < Tmin:
            Tmin = Tbl

        return Tmin

    def log_state(self):
        """
        A helper to append the current model variables into self.states dictionary.
        """
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

        # CO (cardiac output) = HR[beats/min]*SV[ml/beat] = ml/min
        CO = self.HR*self.SV  
        self.states["CO"].append(CO)

        # CI (cardiac index) = CO / SA  => ml/min per m^2
        if self.SA > 0:
            CI = CO/self.SA
        else:
            CI = 0.0
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

    def step(self, inputs1, inputs2, Tnext):
        """
        Translated from ScenarioModel.java step(...)
        The big 2-step integration scheme is here.
        """
        Ta   = inputs1.Ta
        Tmr  = inputs1.Tmr
        Pvap = inputs1.Pvap
        Vair = inputs1.Vair
        Iclo = inputs1.Iclo
        Im   = inputs1.Im

        workMode = inputs2.workMode
        Vmove    = inputs2.Vmove
        self.Mtot= inputs2.Mtot
        Mrst     = inputs2.Mrst
        Mext     = inputs2.Mext
        fluidIntake = inputs2.fluidIntake

        VO2 = self.Mtot/341.0
        self.Mnet= self.Mtot - Mext
        Mmu = self.Mnet - (Mrst*(1.0 - fMrstMu))
        Mmu2= Mmu
        Mtot2= self.Mtot

        EEmuOld= self.EEmu
        EEmuNew= Mmu + Mext
        if EEmuNew!=EEmuOld:
            self.T_EEmuNew= self.time

        # compute new SVmax & BFvskMax
        if VO2<=0.5:
            SVmax=85.0
            self.BFvskMaxNew=7000.0
        elif VO2>=2.0:
            SVmax=130.0
            self.BFvskMaxNew=5000.0
        else:
            SVmax= 30.0*(VO2-0.5)+85.0
            self.BFvskMaxNew=7000.0 - (VO2-0.5)*1333.0

        Mra  = 0.0
        Mcr  = self.MrstCr
        Mfat = self.MrstFat
        Mvsk = self.MrstVsk
        Mski = self.MrstSk

        Eres = self.SA*0.0023*self.Mnet*(44.0 - Pvap)
        Cres = self.SA*0.0012*self.Mnet*(34.0 - Ta)
        RadFact = 1.0 + 0.15*Iclo

        # compute Kc, Kr
        Kc = self.compKc(workMode, Vair, Vmove, Mrst)
        Kr = self.compKr(Kc, Iclo, Tmr, Ta)

        Kop = Kc+Kr
        self.Top = (Kr*Tmr + Kc*Ta)/Kop
        self.ClEffFact = 1.0/(1.0+Kop*KClo*Iclo)

        self.DryHtLoss1 = self.SA*Kop*self.ClEffFact*(self.Tsk-self.Top)
        self.DryHtLoss2 = self.SA*Kop*self.ClEffFact*(self.TTC-self.Top)
        ClPermFact = (Kop/Kc)*self.ClEffFact*Im
        Psk = utils.SatVP(self.Tsk)
        self.Emax= self.SA*Lewis*Kc*(Psk - Pvap)*ClPermFact

        self.compShiverCorrection(Mmu2, Mtot2, Mext, EEmuOld, EEmuNew)

        # The time stepping loop:
        while self.time< Tnext:

            # store Ts and derivs at beginning
            Tra1, Tcr1, Tmu1, Tfat1, Tvsk1, Tsk1, TTC1 = (
                self.Tra, self.Tcr, self.Tmu, self.Tfat, self.Tvsk, self.Tsk, self.TTC)
            dQra_dt1  = self.dQra_dt
            dQcr_dt1  = self.dQcr_dt
            dQmu_dt1  = self.dQmu_dt
            dQfat_dt1 = self.dQfat_dt
            dQvsk_dt1 = self.dQvsk_dt
            dQsk_dt1  = self.dQsk_dt
            dQTC_dt1  = self.dQTC_dt
            dO2debt_dt1 = self.dO2debt_dt
            SR1= self.SR

            # estimate Ts at end of step
            self.Tra  = Tra1  + dQra_dt1 / self.HtCapRA  * self.dt
            self.Tcr  = Tcr1  + dQcr_dt1 / self.HtCapCr  * self.dt
            self.Tmu  = Tmu1  + dQmu_dt1 / self.HtCapMu  * self.dt
            self.Tfat = Tfat1 + dQfat_dt1/ self.HtCapFat * self.dt
            self.Tvsk = Tvsk1 + dQvsk_dt1/ self.HtCapVsk * self.dt
            self.Tsk  = Tsk1  + dQsk_dt1 / self.HtCapSk  * self.dt
            self.TTC  = TTC1  + dQTC_dt1 / self.HtCapSk  * self.dt

            # update T-dependent props with T-est
            self.compStrokeVolume(SVmax)
            self.compCOreq()
            self.compVascularBloodFlow(VO2)
            self.compVskSkCond(workMode)
            self.compMuscleBloodFlow()
            self.compShivering()
            self.compSweatRate()
            self.compDripSkWetEsk(Kop, Kc, Pvap, self.ClEffFact, Im)
            self.compShiverCorrection(Mmu2, Mtot2, Mext, EEmuOld, EEmuNew)
            self.compDryHeatLoss(RadFact, Kc, Tmr, Ta, Iclo)

            # estimate derivative
            self.deriv(Cres, Eres, Mcr, Mfat, Mvsk, Mski)
            dQra_dt2 = self.dQra_dt
            dQcr_dt2 = self.dQcr_dt
            dQmu_dt2 = self.dQmu_dt
            dQfat_dt2= self.dQfat_dt
            dQvsk_dt2= self.dQvsk_dt
            dQsk_dt2 = self.dQsk_dt
            dQTC_dt2 = self.dQTC_dt
            dO2debt_dt2= self.dO2debt_dt
            SR2 = self.SR

            # average
            self.dQra_dt=  0.5*(dQra_dt1 + dQra_dt2)
            self.dQcr_dt=  0.5*(dQcr_dt1 + dQcr_dt2)
            self.dQmu_dt=  0.5*(dQmu_dt1 + dQmu_dt2)
            self.dQfat_dt= 0.5*(dQfat_dt1+ dQfat_dt2)
            self.dQvsk_dt= 0.5*(dQvsk_dt1+ dQvsk_dt2)
            self.dQsk_dt=  0.5*(dQsk_dt1 + dQsk_dt2)
            self.dQTC_dt=  0.5*(dQTC_dt1 + dQTC_dt2)
            self.dO2debt_dt=0.5*(dO2debt_dt1+ dO2debt_dt2)
            self.SR=0.5*(SR1+SR2)

            # final Ts
            self.Tra  = Tra1  + self.dQra_dt / self.HtCapRA  * self.dt
            self.Tcr  = Tcr1  + self.dQcr_dt / self.HtCapCr  * self.dt
            self.Tmu  = Tmu1  + self.dQmu_dt / self.HtCapMu  * self.dt
            self.Tfat = Tfat1 + self.dQfat_dt/ self.HtCapFat * self.dt
            self.Tvsk = Tvsk1 + self.dQvsk_dt/ self.HtCapVsk * self.dt
            self.Tsk  = Tsk1  + self.dQsk_dt / self.HtCapSk  * self.dt
            self.TTC  = TTC1  + self.dQTC_dt / self.HtCapSk  * self.dt

            # heat changes
            self.Qra  += 0.06*self.dQra_dt* self.dt
            self.Qcr  += 0.06*self.dQcr_dt* self.dt
            self.Qmu  += 0.06*self.dQmu_dt* self.dt
            self.Qfat += 0.06*self.dQfat_dt*self.dt
            self.Qvsk += 0.06*self.dQvsk_dt*self.dt
            self.Qsk  += 0.06*self.dQsk_dt* self.dt
            self.Qtot  = (self.Qra+self.Qcr+self.Qmu
                         +self.Qfat+self.Qvsk+self.Qsk)

            # O2 debt, fluid
            self.O2debt += self.dO2debt_dt*self.dt
            self.fluidLoss += (self.SR - fluidIntake)* self.dt

            # next cycle updates
            self.compStrokeVolume(SVmax)
            self.compCOreq()
            self.compVascularBloodFlow(VO2)
            self.compVskSkCond(workMode)
            self.compMuscleBloodFlow()
            self.compShivering()
            self.compSweatRate()
            self.compDripSkWetEsk(Kop, Kc, Pvap, self.ClEffFact, Im)
            self.compShiverCorrection(Mmu2, Mtot2, Mext, EEmuOld, EEmuNew)
            self.compDryHeatLoss(RadFact, Kc, Tmr, Ta, Iclo)
            self.deriv(Cres, Eres, Mcr, Mfat, Mvsk, Mski)

            self.time+= self.dt
            if (self.time+self.FUZZ)> Tnext:
                self.time= Tnext
            self.iter+=1

            # compute dt from stability
            Tmin = self.compTmin()
            if Tmin<=0:
                raise ModelException(self.NEGATIVE_TIMESTEP)
            self.dt= self.stabilityFactor*Tmin
            self.FUZZ= 0.01*self.dt

            # circadian
            timeOfDay= self.time/60.0 + self.startTime
            if timeOfDay>24:
                timeOfDay-=24
            if self.circadianModel:
                # stub
                self.ThypSet= 36.96 + self.delThypSet  # or real interpolation
            self.log_state()

        # At the end:
        # compute PSI
        self.PSI = 5.0*(self.Tcr - self.Tre0)/(39.5 - self.Tre0)\
                 + 5.0*(self.HR - self.HR0)/(180.0 - self.HR0)
        if self.PSI<0.0:
            self.PSI=0.0
        self.compTbdy()
        CrSkdT = self.Tcr- self.Tsk
        RaSkdT = self.Tra- self.Tsk
        MRC = self.Mnet - (Eres + Cres) - self.DryHtLoss1
        dQtot_dt = (self.dQra_dt+self.dQcr_dt+self.dQmu_dt
                    +self.dQfat_dt+self.dQvsk_dt+self.dQsk_dt)

        # store final in self.predict
        self.predict.Tra  = self.Tra
        self.predict.Tcr  = self.Tcr
        self.predict.Tmu  = self.Tmu
        self.predict.Tfat = self.Tfat
        self.predict.Tvsk = self.Tvsk
        self.predict.Tsk  = self.Tsk
        self.predict.Tcl  = self.Tcl
        self.predict.Tbdy = self.Tbdy

        self.predict.BFra  = self.BFra
        self.predict.BFcr  = self.BFcr
        self.predict.BFmu  = self.BFmu
        self.predict.BFfat = self.BFfat
        self.predict.BFvsk = self.BFvsk

        self.predict.dQtot_dt= dQtot_dt/self.SA
        self.predict.Qtot = self.Qtot
        self.predict.Qra  = self.Qra
        self.predict.Qcr  = self.Qcr
        self.predict.Qmu  = self.Qmu
        self.predict.Qfat = self.Qfat
        self.predict.Qvsk = self.Qvsk
        self.predict.Qsk  = self.Qsk

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

        # setPts stuff
        self.predict.ThypSet  = self.ThypSet
        self.predict.TimeOfDay= timeOfDay

        self.log_state()

    def exit(self):
        """
        Java: exit() - any final clean up
        """
        # no real file logic here in Python
        pass

def main():
    """
    Minimal demonstration with visualization:
    """

    segments = {
        'time'          : [60],
        "BW"            : [77.61],
        "SA"            : [1.8],
        "AGE"           : [44],
        "Ta"            : [30.0],
        "Tmr"           : [35.0],
        "Vair"          : [0.1],
        "Pvap"          : [25.0],
        "Iclo"          : [0.5],
        "Im"            : [0.5],
        "PctFat"        : [20.0],
        "Mtot"          : [400.0],
        "Mrst"          : [100.0],
        "Mext"          : [0],
        "Vmove"         : [1.0],
        "workMode"      : ['f'],
        "fluidIntake"   : [0.0],
        "acclimIndex"   : [NO_ACCLIMATION],
        "dehydIndex"    : [DEHYD_NORMAL],
        "startTime"     : [0.0],
        "Tcr0"          : [37.06],
        "circadianModel": [False],
        "tcoreOverride" : [True]
    }
    model = ScenarioModel(segments)

    # time series
    time = model.states["time"]
    Tcr = model.states["Tcr"]
    HR = model.states["HR"]
    FluidLoss = model.states["fluidLoss"]
    Tsk = model.states["Tsk"]

    # create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # plot core temperature
    axs[0, 0].plot(time, Tcr, label='Core Temperature (Tcr)')
    axs[0, 0].set_xlabel('Time (min)')
    axs[0, 0].set_ylabel('Temperature (C)')
    axs[0, 0].set_title('Core Temperature Over Time')
    axs[0, 0].legend()

    # plot heart rate
    axs[0, 1].plot(time, HR, label='Heart Rate (HR)', color='orange')
    axs[0, 1].set_xlabel('Time (min)')
    axs[0, 1].set_ylabel('Heart Rate (bpm)')
    axs[0, 1].set_title('Heart Rate Over Time')
    axs[0, 1].legend()

    # plot fluid loss
    axs[1, 0].plot(time, FluidLoss, label='Fluid Loss', color='green')
    axs[1, 0].set_xlabel('Time (min)')
    axs[1, 0].set_ylabel('Fluid Loss (g)')
    axs[1, 0].set_title('Fluid Loss Over Time')
    axs[1, 0].legend()

    # plot Tsk
    axs[1, 1].plot(time, Tsk, label='Tsk', color='red')
    axs[1, 1].set_xlabel('Time (min)')
    axs[1, 1].set_ylabel('Tsk')
    axs[1, 1].set_title('Tsk Over Time')
    axs[1, 1].legend()

    # adjust layout
    plt.tight_layout()
    plt.show()

    # save the figure to the root folder
    fig.savefig('/opt/biogears/core/build/runtime/scenario_python/scenario_model_output.png')

    # show final results
    p = model.getPredict()
    print("=== Final Predictions ===")
    print(f"  Tcr = {p.Tcr:.2f} C")
    print(f"  Tsk = {p.Tsk:.2f} C")
    print(f"  HR  = {p.HR:.2f} bpm")
    print(f"  PSI = {p.PSI:.2f}")
    print(f"  FluidLoss= {p.FluidLoss:.2f} g")

if __name__ == "__main__":
    main()
