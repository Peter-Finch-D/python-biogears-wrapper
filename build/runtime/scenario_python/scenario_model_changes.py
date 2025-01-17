"""
scenario_model.py

A Python module that implements the logic from ScenarioModel.java as closely
as possible, using the Python 'scenario_constants', 'tissue_conductance', and 'utils'
modules. 

Note: This is quite a large code translation from Java, with a 2-step integration
scheme in step(). 
"""

import math
import sys

from scenario_constants import (
    # Constants:
    # scenario_constants includes many from ScenarioConstants.java
    NO_ACCLIMATION, PART_ACCLIMATION, FULL_ACCLIMATION,
    DEHYD_NORMAL, DEHYD_MODERATE, DEHYD_SEVERE,
    fMrstCr, fMrstMu, fMrstFat, fMrstVsk, fMrstSk,
    fCOrstCr, fCOrstMu, fCOrstFat, fCOrstVsk,
    SpHtCr, SpHtMu, SpHtFat, SpHtSk, SpHtBl,
    Lewis, Boltz, SArad, KClo, HtVap,
    # Some Java-esque constants we replicate:
    PI,
    # etc. as needed
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
        self.BFmuReq  = 0.0
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

    def getPredict(self):
        return self.predict

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

    def init(self, input_data):
        """
        Accepts a dictionary with the following fields:
            body_weight
            height
            age
            ambient_temperature
            mean_radiant_temperature
            relative_humidity
            wind_speed
            clo
            im_over_clo
            cloVg
            im_over_cloVg
            metabolic_rate
            resting_metabolic_rate
            resting_heart_rate
            initial_skin_temperature
            initial_core_temperature
        """
        # Map inputs
        self.BW     = input_data['body_weight']
        self.height = input_data['height']
        age = input_data['age']
        Ta  = input_data['ambient_temperature']
        Tmr = input_data['mean_radiant_temperature']
        rh  = input_data['relative_humidity']
        Vair = input_data['wind_speed']
        Iclo = input_data['clo']
        Im   = input_data['im_over_clo']
        cloVg   = input_data['cloVg']
        Im_cloVg = input_data['im_over_cloVg']
        self.Mtot = input_data['metabolic_rate']
        Mrst = input_data['resting_metabolic_rate']
        self.HRrst = input_data['resting_heart_rate']
        self.Tsk  = input_data['initial_skin_temperature']
        self.Tcr  = input_data['initial_core_temperature']

        # We retain the old logic that used self.PctFat, SA, etc. 
        # For height-based surface area, you could approximate it or supply it separately.
        # Here we'll do a simple approach:
        # Example for surface area from DuBois formula:
        self.SA = 0.20247 * (self.height**0.725) * (self.BW**0.425)

        # For partial vapor pressure from relative humidity:
        # Suppose we approximate saturate vapor pressure at Ta to get Pvap
        # This is just a placeholder or example:
        Pvap = 42.0 * rh  # simplistic example for demonstration
        self.PctFat = 15.0  # or set from dictionary if desired

        # The old code's Mnet, TcoreOverride, etc. could also be adapted:
        Mext = 20.0
        self.Mnet = self.Mtot - Mext
        workMode = 'r'  # or adapt if needed

        # We remove references to old Inputs1, Inputs2, Inputs3 and just proceed:
        # ...existing code from ScenarioModel.init(...) except referencing input_data...

        tissueConductance = TissueConductance()
        # We'll define the dictionary approach for TissueConductance if needed
        # or just call compute with the minimal placeholders:
        class _TissueInputs:
            def __init__(self, Ta, Vair, PctFat):
                self.Ta = Ta
                self.Vair = Vair
                self.PctFat = PctFat
                self.BW = self.BW if hasattr(self, 'BW') else 70.0
                self.SA = self.SA if hasattr(self, 'SA') else 1.8

        ti = _TissueInputs(Ta, Vair, self.PctFat)
        tissueConductance.compute(ti)

        self.CrMuCond   = tissueConductance.getCrMuCond()
        self.MuFatCond  = tissueConductance.getMuFatCond()
        self.FatVskCond = tissueConductance.getFatVskCond()
        self.VskSkCond  = tissueConductance.getVskSkCond()
        self.SkCondConst= tissueConductance.getSkCondConst()

        # ...existing code for heat capacities, compKc, compKr, etc....
        Kc = self.compKc(workMode, Vair, 1.0, Mrst)
        Kr = self.compKr(Kc, Iclo, Tmr, Ta)
        Kop = Kr + Kc
        self.Top = (Kr * Tmr + Kc * Ta) / Kop
        self.ClEffFact = 1.0 / (1.0 + Kop * KClo * Iclo)
        self.Tcl = self.Top + (self.ClEffFact*(self.Tsk - self.Top))

        # Simplify Eres, Cres, etc. for demonstration:
        Eres = self.SA * 0.0023 * self.Mnet * (44.0 - Pvap)
        Cres = self.SA * 0.0012 * self.Mnet * (34.0 - Ta)
        self.deriv(Cres, Eres, 0.0, 0.0, 0.0, 0.0)

        # Set up stroke volume, etc.:
        SVmax = self.COrst / self.HRrst if self.HRrst != 0 else 70
        self.SVold = SVmax
        # ...existing code that sets up final states in init...

        self.iter = 0
        self.dt = 0.025
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

    def step(self, input_data, Tnext):
        # Replace references to Inputs1,2 with input_data
        # ...existing step logic, but fetch what you need from input_data if needed...
        # e.g. Ta = input_data['ambient_temperature']
        self.time += Tnext
        self.states["time"].append(self.time)
        # ...existing code that updates states...
        pass

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
    input_data = {
        'body_weight': 70.0,
        'height': 1.75,
        'age': 30,
        'ambient_temperature': 30.0,
        'mean_radiant_temperature': 30.0,
        'relative_humidity': 0.5,
        'wind_speed': 0.5,
        'clo': 0.5,
        'im_over_clo': 1.0,
        'cloVg': 0.4,
        'im_over_cloVg': 1.2,
        'metabolic_rate': 500.0,
        'resting_metabolic_rate': 100.0,
        'resting_heart_rate': 70.0,
        'initial_skin_temperature': 33.0,
        'initial_core_temperature': 37.2
    }
    model = ScenarioModel()
    model.init(input_data)
    model.step(input_data, 30.0)

    # time series
    time = model.states["time"]
    Tcr = model.states["Tcr"]
    HR = model.states["HR"]
    FluidLoss = model.states["fluidLoss"]
    Tsk = model.states["Tsk"]
    """
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
    """
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
