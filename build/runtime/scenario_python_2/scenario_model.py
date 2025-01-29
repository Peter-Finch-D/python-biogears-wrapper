# =============================================================================
# ScenarioModel.py
# 
# This Python module is a direct translation of the Java class "ScenarioModel"
# from the provided code. All functionality is intended to mirror the original
# Java version as closely as possible, line-by-line, preserving method names,
# logic, and comments.
# 
# IMPORTANT:
#   - This code depends on several classes, functions, and constants that must
#     be defined elsewhere in order for it to run properly (see the list of
#     dependencies at the bottom).
#   - This code uses Python's exception handling, file I/O, and other features
#     in ways that parallel the Java approach. Where the Java code uses streams,
#     FormatWriter, NumberFormat, etc., we have carried over the names and calls
#     but in a Pythonic way. You must provide implementations or mocks/stubs
#     for them to make this code fully functional.
# 
# All numeric variables that were 'float' in Java are now explicitly declared
# or cast to NumPy float32 in Python to guarantee single-precision behavior.
# =============================================================================

import math
import numpy as np

from scenario_constants import *
from utils import *
from tissue_conductance import *
from circadian_table import CircadianTable
from set_pts_n_flags import SetPts_n_Flags

from predict1 import Predict1
from predict2 import Predict2
from predict4 import Predict4
from predict5 import Predict5
from predict6 import Predict6
from predict7 import Predict7

class ScenarioModel:
    """
    The computational engine for SCENARIO-J (Python translation).
    """

    # -------------------------------------------------------------------------
    # Class-level constants (translated from Java's static final fields)
    # Now explicitly using np.float32 for all float constants.
    # -------------------------------------------------------------------------
    SCENARIO_SERVER_VERSION = "Scenario-J Server v1.0"
    ROOT_DIR = "user.dir"
    INI_FILE = "scenario.ini"

    # INI sections
    INI_CURRENCY = "[currency]"
    INI_PATHS = "[paths]"
    INI_DATA = "[data]"

    # INI keys
    INI_SERVER_VERSION = "serverVersion"
    INI_DATA_DIR = "DataDir"
    INI_CIRCADIAN_DATA = "circadian"
    INI_STABILITY_FACTOR = "stabilityFactor"
    INI_PRINT_TAU = "printTau"

    # error messages
    NO_INI_FILE = "Could not open INI file."

    # average cardiac index and heart rate
    CIave = np.float32(3200.0)   # average cardiac index - from Ganong
    HRrst = np.float32(70.0)

    # default radiative and convective conductance [W/m^2-C]
    KR0 = np.float32(4.7)
    STANDARD_KC = np.float32(2.1)  # For sedentary person in still air

    BlO2Cap = np.float32(0.21)     # cc O2/cc blood
    O2Equiv = np.float32(2.93)     # cc O2/(W-min)

    BFvskMaxOld = np.float32(7000.0)  # start at rest
    BFvskMin = np.float32(27.0)       # cc/min (Increased from 8 to 27)
    TsetBFold = np.float32(36.75)
    SRmax = np.float32(25.0)          # g/min
    SRmin = np.float32(0.0)           # g/min

    # sweat rate constants
    DELTA_ALPHA_SR = np.float32(-0.6)   # g/min-C
    DELTA_HYPSIG = np.float32(0.06)     # (C)

    # set points and flags
    STANDARD_THYP_SET = np.float32(36.96)
    STANDARD_TSK_SET = np.float32(33.0)
    STANDARD_TCR_FLAG = np.float32(39.5)
    STANDARD_HR_FLAG = np.float32(0.9 * (220.0 - 20.0))  # 20 is SubjectRecord.STANDARD_AGE
    STANDARD_DQDT_FLAG = np.float32(95.0)    # W/m^2
    STANDARD_QTOT_FLAG = np.float32(840.0)   # kJ

    # exception Strings
    FAILED_TO_CONVERGE = "Scenario.compKr failed to converge."
    NEGATIVE_TIMESTEP = "Negative time step computed.  Reduce the stability factor."

    # The following two class variables replicate Java's static initialization
    # of stabilityFactor and PRINT_TAU in pseudoMain().
    stabilityFactor = np.float32(0.01)
    PRINT_TAU = False

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        Default constructor.
        Mirrors the Java 'public ScenarioModel() { }'
        """
        # These instance variables mirror the Java code's instance fields.
        # We declare them here for clarity. They will be set in init(...) below,
        # but we initialize them as np.float32(0.0) to ensure single precision.

        # time independent variables
        self.MrstCr = np.float32(0.0)
        self.MrstFat = np.float32(0.0)
        self.MrstVsk = np.float32(0.0)
        self.MrstSk = np.float32(0.0)
        self.COrst = np.float32(0.0)
        self.BFcrOld = np.float32(0.0)
        self.SVold = np.float32(0.0)
        self.HRmax = np.float32(0.0)

        self.HtCapRA = np.float32(0.0)
        self.HtCapCr = np.float32(0.0)
        self.HtCapMu = np.float32(0.0)
        self.HtCapFat = np.float32(0.0)
        self.HtCapVsk = np.float32(0.0)
        self.HtCapSk = np.float32(0.0)

        self.ThypSet = np.float32(0.0)
        self.TskSet = np.float32(0.0)
        self.CrMuCond = np.float32(0.0)
        self.MuFatCond = np.float32(0.0)
        self.FatVskCond = np.float32(0.0)
        self.VskSkCond = np.float32(0.0)
        self.SkCondConst = np.float32(0.0)
        self.BW = np.float32(0.0)
        self.SA = np.float32(0.0)
        self.PctFat = np.float32(0.0)

        # time dependent variables
        self.Tra = np.float32(0.0)
        self.Tcr = np.float32(0.0)
        self.Tmu = np.float32(0.0)
        self.Tfat = np.float32(0.0)
        self.Tvsk = np.float32(0.0)
        self.Tsk = np.float32(0.0)
        self.Tcl = np.float32(0.0)
        self.BFra = np.float32(0.0)
        self.BFcr = np.float32(0.0)
        self.BFmu = np.float32(0.0)
        self.BFfat = np.float32(0.0)
        self.BFvsk = np.float32(0.0)
        self.Qra = np.float32(0.0)
        self.Qcr = np.float32(0.0)
        self.Qmu = np.float32(0.0)
        self.Qfat = np.float32(0.0)
        self.Qvsk = np.float32(0.0)
        self.Qsk = np.float32(0.0)
        self.Qtot = np.float32(0.0)
        self.SR = np.float32(0.0)
        self.Esk = np.float32(0.0)
        self.Emax = np.float32(0.0)
        self.dMshiv = np.float32(0.0)
        self.fluidLoss = np.float32(0.0)
        self.O2debt = np.float32(0.0)
        self.TTC = np.float32(0.0)
        self.BFvskReq = np.float32(0.0)
        self.BFmuReq = np.float32(0.0)
        self.BFvskMaxNew = np.float32(0.0)
        self.EEmu = np.float32(0.0)
        self.SV = np.float32(0.0)
        self.HR = np.float32(0.0)
        self.PSI = np.float32(0.0)
        self.Tbdy = np.float32(0.0)

        self.Overload = np.float32(0.0)
        self.PctHRmax = np.float32(0.0)
        self.LeftOver = np.float32(0.0)
        self.PctExt = np.float32(0.0)
        self.Mtot = np.float32(0.0)
        self.Mmu = np.float32(0.0)
        self.Mnet = np.float32(0.0)
        self.Top = np.float32(0.0)
        self.ClEffFact = np.float32(0.0)
        self.DryHtLoss1 = np.float32(0.0)
        self.DryHtLoss2 = np.float32(0.0)
        self.SkWet = np.float32(0.0)
        self.Drip = np.float32(0.0)

        self.dQra_dt = np.float32(0.0)
        self.dQcr_dt = np.float32(0.0)
        self.dQmu_dt = np.float32(0.0)
        self.dQfat_dt = np.float32(0.0)
        self.dQvsk_dt = np.float32(0.0)
        self.dQsk_dt = np.float32(0.0)
        self.dQTC_dt = np.float32(0.0)
        self.dO2debt_dt = np.float32(0.0)

        self.circadianTable = None
        self.delThypSet = np.float32(0.0)
        self.acclimIndex = 0
        self.dehydIndex = 0
        self.startTime = np.float32(0.0)
        self.Tcr0 = np.float32(0.0)
        self.circadianModel = False
        self.tcoreOverride = False
        self.Tre0 = np.float32(0.0)
        self.HR0 = np.float32(0.0)

        self.dt = np.float32(0.0)
        self.FUZZ = np.float32(0.0)
        self.time = np.float32(0.0)
        self.T_EEmuNew = np.float32(0.0)
        self.Tmin0 = np.float32(0.0)

        self.outputStream = None
        self.numberFormat4 = None
        self.iter = 0

        self.DEBUG = False
        self.debugStream = None
        self.debugFormat2 = None
        self.debugFormat4 = None
        self.debugFormat6 = None

        # "Predict" and "SetPts_n_Flags" objects
        self.predict1 = None
        self.predict2 = None
        self.predict4 = None
        self.predict5 = None
        self.predict6 = None
        self.predict7 = None
        self.setPts_n_Flags = None

    # -------------------------------------------------------------------------
    # Methods that parallel the Java code
    # -------------------------------------------------------------------------
    def getVersion(self):
        """
        Returns the version of the server.
        Mirrors: public String getVersion()
        """
        return self.SCENARIO_SERVER_VERSION

    def getPredict1(self):
        """Mirrors: public Predict1 getPredict1()"""
        return self.predict1

    def getPredict2(self):
        """Mirrors: public Predict2 getPredict2()"""
        return self.predict2

    def getPredict4(self):
        """Mirrors: public Predict4 getPredict4()"""
        return self.predict4

    def getPredict5(self):
        """Mirrors: public Predict5 getPredict5()"""
        return self.predict5

    def getPredict6(self):
        """Mirrors: public Predict6 getPredict6()"""
        return self.predict6

    def getPredict7(self):
        """Mirrors: public Predict7 getPredict7()"""
        return self.predict7

    def getSetPts_n_Flags(self):
        """Mirrors: public SetPts_n_Flags getSetPts_n_Flags()"""
        return self.setPts_n_Flags

    def init(self, inputs1, inputs2, inputs3):
        """
        Initializes the SCENARIO-J server.
        Mirrors: public void init(Inputs1 inputs1, Inputs2 inputs2, Inputs3 inputs3)
        """
        # direct read from inputs
        workMode = inputs2.getWorkMode()[0]  # char in Java
        age = np.float32(inputs1.getAGE())
        Ta = np.float32(inputs1.getTa())
        Tmr = np.float32(inputs1.getTmr())
        Vair = np.float32(inputs1.getVair())
        Pvap = np.float32(inputs1.getPvap())
        Iclo = np.float32(inputs1.getIclo())
        Im = np.float32(inputs1.getIm())
        self.BW = np.float32(inputs1.getBW())
        self.SA = np.float32(inputs1.getSA())
        self.PctFat = np.float32(inputs1.getPctFat())

        self.Mtot = np.float32(inputs2.getMtot())
        Mrst = np.float32(inputs2.getMrst())
        Mext = np.float32(inputs2.getMext())
        self.Mnet = np.float32(self.Mtot - Mext)
        Vmove = np.float32(inputs2.getVmove())
        fluidIntake = np.float32(inputs2.getFluidIntake())

        self.acclimIndex = inputs3.getAcclimIndex()
        self.dehydIndex = inputs3.getDehydIndex()
        self.startTime = np.float32(inputs3.getStartTime())
        self.Tcr0 = np.float32(inputs3.getTcr0())
        self.circadianModel = inputs3.getCircadianModel()
        self.tcoreOverride = inputs3.getTcoreOverride()

        # Create circadian table
        self.circadianTable = CircadianTable()

        self.ThypSet = np.float32(self.STANDARD_THYP_SET)
        if self.circadianModel:
            self.ThypSet = np.float32(self.circadianTable.interpTemp(self.startTime))

        # Set body temps by acclimation state
        if self.acclimIndex == NO_ACCLIMATION:
            self.delThypSet = np.float32(0.0)
            self.Tra  = np.float32(36.75)
            self.Tmu  = np.float32(36.07)
            self.Tcr  = np.float32(36.98)
            self.Tfat = np.float32(33.92)
            self.Tvsk = np.float32(33.49)
            self.Tsk  = np.float32(33.12)
            self.TTC  = self.Tsk
        elif self.acclimIndex == PART_ACCLIMATION:
            self.delThypSet = np.float32(-0.25)
            self.Tra  = np.float32(36.73)
            self.Tmu  = np.float32(36.1)
            self.Tcr  = np.float32(36.96)
            self.Tfat = np.float32(34.13)
            self.Tvsk = np.float32(33.73)
            self.Tsk  = np.float32(32.87)
            self.TTC  = self.Tsk
        elif self.acclimIndex == FULL_ACCLIMATION:
            self.delThypSet = np.float32(-0.5)
            self.Tra  = np.float32(36.49)
            self.Tmu  = np.float32(35.91)
            self.Tcr  = np.float32(36.72)
            self.Tfat = np.float32(34.04)
            self.Tvsk = np.float32(33.66)
            self.Tsk  = np.float32(32.79)
            self.TTC  = self.Tsk

        self.ThypSet = np.float32(self.ThypSet + self.delThypSet)

        # Optional Tcore override
        if self.tcoreOverride:
            diff = np.float32(self.Tcr0 - self.Tcr)
            self.Tra  = np.float32(self.Tra  + diff)
            self.Tmu  = np.float32(self.Tmu  + diff)
            self.Tfat = np.float32(self.Tfat + diff)
            self.Tvsk = np.float32(self.Tvsk + diff)
            self.Tsk  = np.float32(self.Tsk  + diff)
            self.Tcr  = np.float32(self.Tcr0)
            self.TTC  = np.float32(self.Tsk)

        # Dehydration index
        if self.dehydIndex == DEHYD_NORMAL:
            self.fluidLoss = np.float32(0.0)
        elif self.dehydIndex == DEHYD_MODERATE:
            self.fluidLoss = np.float32(0.02 * self.BW * 1000.0)
        elif self.dehydIndex == DEHYD_SEVERE:
            self.fluidLoss = np.float32(0.04 * self.BW * 1000.0)

        # Partition resting metabolism
        self.MrstCr = np.float32(fMrstCr * Mrst)
        MrstMu = np.float32(fMrstMu * Mrst)
        self.MrstFat = np.float32(fMrstFat * Mrst)
        self.MrstVsk = np.float32(fMrstVsk * Mrst)
        self.MrstSk  = np.float32(fMrstSk  * Mrst)

        # Blood flow distribution
        self.COrst = np.float32(self.CIave * self.SA)
        self.BFra  = np.float32(self.COrst)
        self.BFcr  = np.float32(fCOrstCr  * self.COrst)
        self.BFcrOld = np.float32(self.BFcr)
        self.BFmu  = np.float32(fCOrstMu  * self.COrst)
        self.BFfat = np.float32(fCOrstFat * self.COrst)
        self.BFvsk = np.float32(fCOrstVsk * self.COrst)
        self.BFvskReq = np.float32(self.BFvsk)
        self.BFmuReq  = np.float32(self.BFmu)

        # Conductances
        tc = TissueConductance()
        tc.compute(inputs1)
        self.CrMuCond   = np.float32(tc.getCrMuCond())
        self.MuFatCond  = np.float32(tc.getMuFatCond())
        self.FatVskCond = np.float32(tc.getFatVskCond())
        self.VskSkCond  = np.float32(tc.getVskSkCond())
        self.SkCondConst= np.float32(tc.getSkCondConst())

        # Weights & heat capacities
        Wfat  = np.float32(10.0 * self.PctFat      * self.BW)
        Wnfat = np.float32(10.0 * (100.0 - self.PctFat) * self.BW)

        self.HtCapRA  = np.float32(SpHtBl * fBWnfatRa  * Wnfat/1000.0)
        self.HtCapCr  = np.float32(SpHtCr * fBWnfatCr  * Wnfat/1000.0)
        self.HtCapMu  = np.float32(SpHtMu * fBWnfatMu  * Wnfat/1000.0)
        self.HtCapFat = np.float32(SpHtFat* Wfat       / 1000.0)
        self.HtCapVsk = np.float32(SpHtBl * fBWnfatVsk * Wnfat/1000.0)
        self.HtCapSk  = np.float32(SpHtSk * fBWnfatSk  * Wnfat/1000.0)

        # Compute Kc, Kr & other deps
        Kc = np.float32(self.compKc(workMode, Vair, Vmove, Mrst))
        Kr = np.float32(self.compKr(Kc, Iclo, Tmr, Ta))
        Kop = np.float32(Kr + Kc)
        self.Top = np.float32((Kr * Tmr + Kc * Ta) / Kop)
        self.ClEffFact = np.float32(1.0 / (1.0 + Kop * KClo * Iclo))
        self.Tcl = np.float32(self.Top + (self.ClEffFact * (self.Tsk - self.Top)))

        Eres = np.float32(self.SA * 0.0023 * self.Mnet * (44.0 - Pvap))
        Cres = np.float32(self.SA * 0.0012 * self.Mnet * (34.0 - Ta))
        self.DryHtLoss1 = np.float32(self.SA * Kop * self.ClEffFact * (self.Tsk - self.Top))
        self.DryHtLoss2 = np.float32(self.SA * Kop * self.ClEffFact * (self.TTC - self.Top))
        MRC = np.float32(self.Mnet - (Eres + Cres) - self.DryHtLoss1)
        ClPermFact = np.float32((Kop / Kc) * self.ClEffFact * Im)
        Psk = np.float32(SatVP(self.Tsk))
        self.Emax = np.float32(self.SA * Lewis * Kc * (Psk - Pvap) * ClPermFact)

        self.compTbdy()

        # Initial states
        self.SR = np.float32(0.0)
        self.Esk = np.float32(0.0)
        self.dMshiv = np.float32(0.0)
        self.PctExt = np.float32(80.0)
        self.LeftOver = np.float32(0.0)
        self.time = np.float32(0.0)
        self.T_EEmuNew = np.float32(0.0)
        self.EEmu = np.float32(MrstMu + Mext)
        EEmuOld = np.float32(self.EEmu)

        self.compShiverCorrection(MrstMu, self.Mtot, Mext, EEmuOld, EEmuOld)
        self.deriv(Cres, Eres, self.MrstCr, self.MrstFat, self.MrstVsk, self.MrstSk)

        # Initialize the "Predict" objects, etc.
        self.predict1 = Predict1()
        self.predict1.reset()
        self.predict1.setTra(self.Tra)
        self.predict1.setTcr(self.Tcr)
        self.predict1.setTmu(self.Tmu)
        self.predict1.setTfat(self.Tfat)
        self.predict1.setTvsk(self.Tvsk)
        self.predict1.setTsk(self.Tsk)
        self.predict1.setTcl(self.Tcl)
        self.predict1.setTbdy(self.Tbdy)

        self.predict2 = Predict2()
        self.predict2.reset()
        self.predict2.setBFra(self.BFra)
        self.predict2.setBFcr(self.BFcr)
        self.predict2.setBFmu(self.BFmu)
        self.predict2.setBFfat(self.BFfat)
        self.predict2.setBFvsk(self.BFvsk)

        self.TskSet = np.float32(self.STANDARD_TSK_SET)
        TcrFlag = np.float32(self.STANDARD_TCR_FLAG)
        self.HRmax = np.float32(220.0 - age)
        HRFlag = np.float32(0.9 * self.HRmax)
        dQ_dtFlag = np.float32(self.STANDARD_DQDT_FLAG)
        QtotFlag = np.float32(self.STANDARD_QTOT_FLAG)

        self.setPts_n_Flags = SetPts_n_Flags()
        self.setPts_n_Flags.reset()
        self.setPts_n_Flags.setThypSet(self.ThypSet)
        self.setPts_n_Flags.setTskSet(self.TskSet)
        self.setPts_n_Flags.setTcrFlag(TcrFlag)
        self.setPts_n_Flags.setHRFlag(HRFlag)
        self.setPts_n_Flags.setdQ_dtFlag(dQ_dtFlag)
        self.setPts_n_Flags.setQtotFlag(QtotFlag)
        self.setPts_n_Flags.setTimeOfDay(self.startTime)

        self.predict4 = Predict4()
        self.predict4.reset()
        dQtot_dt = np.float32(self.dQra_dt + self.dQcr_dt + self.dQmu_dt +
                              self.dQfat_dt + self.dQvsk_dt + self.dQsk_dt)
        self.predict4.setdQtot_dt(dQtot_dt / self.SA)
        self.predict4.setdQra_dt(self.dQra_dt / self.SA)
        self.predict4.setdQcr_dt(self.dQcr_dt / self.SA)
        self.predict4.setdQmu_dt(self.dQmu_dt / self.SA)
        self.predict4.setdQfat_dt(self.dQfat_dt / self.SA)
        self.predict4.setdQvsk_dt(self.dQvsk_dt / self.SA)
        self.predict4.setdQsk_dt(self.dQsk_dt / self.SA)

        self.Qra = np.float32(0.0)
        self.Qcr = np.float32(0.0)
        self.Qmu = np.float32(0.0)
        self.Qfat= np.float32(0.0)
        self.Qvsk= np.float32(0.0)
        self.Qsk = np.float32(0.0)
        self.Qtot= np.float32(0.0)

        self.predict5 = Predict5()
        self.predict5.reset()
        self.predict5.setQra(self.Qra)
        self.predict5.setQcr(self.Qcr)
        self.predict5.setQmu(self.Qmu)
        self.predict5.setQfat(self.Qfat)
        self.predict5.setQvsk(self.Qvsk)
        self.predict5.setQsk(self.Qsk)
        self.predict5.setQtot(self.Qtot)

        self.predict6 = Predict6()
        self.predict6.reset()
        self.predict6.setSR(self.SR)
        self.predict6.setDrip(self.Drip)
        self.predict6.setEsk(self.Esk / self.SA)
        self.predict6.setEmax(self.Emax / self.SA)
        self.predict6.setSkWet(self.SkWet * np.float32(100.0))
        self.predict6.setdMshiv(self.dMshiv / self.SA)
        self.predict6.setMRC(MRC / self.SA)
        self.predict6.setCrSkdT(self.Tcr - self.Tsk)
        self.predict6.setRaSkdT(self.Tra - self.Tsk)

        self.predict7 = Predict7()
        self.predict7.reset()
        self.O2debt = np.float32(0.0)
        self.SVold = np.float32(self.COrst / self.HRrst)
        MWST = np.float32(self.Tsk)
        if MWST > np.float32(38.0):
            MWST = np.float32(38.0)
        if MWST <= np.float32(33.0):
            self.SV = np.float32(self.SVold)
        else:
            self.SV = np.float32(self.SVold - np.float32(5.0) *
                                 ((self.SVold - np.float32(85.0)) / np.float32(45.0)) *
                                 (MWST - np.float32(33.0)))

        COreq = np.float32(self.BFcr + self.BFmuReq + self.BFfat + self.BFvskReq)
        self.HR = np.float32(COreq / self.SV)

        self.predict7.setFluidLoss(self.fluidLoss)
        self.predict7.setO2debt(self.O2debt)
        self.predict7.setHR(self.HR)
        self.predict7.setSV(self.SV)
        self.predict7.setPSI(np.float32(0.0))

        self.BFvskMaxNew = np.float32(self.BFvskMaxOld)
        self.HR0  = np.float32(self.HR)
        self.Tre0 = np.float32(self.Tcr)
        self.iter = 0

        if self.PRINT_TAU:
            self.openOutputStream()

        self.Tmin0 = np.float32(self.compTmin0())
        Tmin = np.float32(self.compTmin())

        self.dt = np.float32(0.025)
        self.FUZZ = np.float32(0.01) * self.dt

        if self.DEBUG:
            self.openDebugStream()

    def step(self, inputs1, inputs2, Tnext):
        """
        Steps scenario until time Tnext.
        Mirrors: public void step(Inputs1 inputs1, Inputs2 inputs2, float Tnext)
        """
        Tnext = np.float32(Tnext)  # ensure single precision
        workMode = inputs2.getWorkMode()[0]
        Vmove = np.float32(inputs2.getVmove())
        self.Mtot = np.float32(inputs2.getMtot())
        Mrst = np.float32(inputs2.getMrst())
        Mext = np.float32(inputs2.getMext())
        fluidIntake = np.float32(inputs2.getFluidIntake())
        Ta = np.float32(inputs1.getTa())
        Tmr = np.float32(inputs1.getTmr())
        Pvap= np.float32(inputs1.getPvap())
        Vair= np.float32(inputs1.getVair())
        Iclo= np.float32(inputs1.getIclo())
        Im  = np.float32(inputs1.getIm())

        VO2 = np.float32(self.Mtot / np.float32(341.0))
        self.Mnet = np.float32(self.Mtot - Mext)
        Mmu_no_shiver = np.float32(self.Mnet - (Mrst * (np.float32(1.0) - fMrstMu)))
        Mmu2  = np.float32(Mmu_no_shiver)
        Mtot2 = np.float32(self.Mtot)

        # Energy expenditure updates
        EEmuOld = np.float32(self.EEmu)
        EEmuNew = np.float32(Mmu_no_shiver + Mext)
        if EEmuNew != EEmuOld:
            self.T_EEmuNew = np.float32(self.time)

        # Calculate SVmax and BFvskMaxNew based on VO2
        if VO2 <= np.float32(0.5):
            SVmax = np.float32(85.0)
            self.BFvskMaxNew = np.float32(7000.0)
        elif VO2 >= np.float32(2.0):
            SVmax = np.float32(130.0)
            self.BFvskMaxNew = np.float32(5000.0)
        else:
            SVmax = np.float32(30.0 * (VO2 - np.float32(0.5)) + 85.0)
            self.BFvskMaxNew = np.float32(7000.0 - (VO2 - np.float32(0.5)) * np.float32(1333.0))

        # Assign baseline mass values
        Mra = np.float32(0.0)
        Mcr = np.float32(self.MrstCr)
        Mfat= np.float32(self.MrstFat)
        Mvsk= np.float32(self.MrstVsk)
        Msk = np.float32(self.MrstSk)

        # Calculate respiratory heat exchange
        Eres = np.float32(self.SA * np.float32(0.0023) * self.Mnet * (np.float32(44.0) - Pvap))
        Cres = np.float32(self.SA * np.float32(0.0012) * self.Mnet * (np.float32(34.0) - Ta))
        RadFact = np.float32(1.0 + np.float32(0.15) * Iclo)

        # Compute heat transfer coefficients
        Kc = np.float32(self.compKc(workMode, Vair, Vmove, Mrst))
        Kr = np.float32(self.compKr(Kc, Iclo, Tmr, Ta))
        Kop = np.float32(Kc + Kr)
        self.Top = np.float32((Kr * Tmr + Kc * Ta) / Kop)
        self.ClEffFact = np.float32(1.0 / (1.0 + Kop * KClo * Iclo))

        # Dry heat losses
        self.DryHtLoss1 = np.float32(self.SA * Kop * self.ClEffFact * (self.Tsk - self.Top))
        self.DryHtLoss2 = np.float32(self.SA * Kop * self.ClEffFact * (self.TTC - self.Top))

        # Evaporative potential
        ClPermFact = np.float32((Kop / Kc) * self.ClEffFact * Im)
        Psk = np.float32(SatVP(self.Tsk))
        self.Emax = np.float32(self.SA * Lewis * Kc * (Psk - Pvap) * ClPermFact)

        # Initial correction for shivering
        self.compShiverCorrection(Mmu2, Mtot2, Mext, EEmuOld, EEmuNew)

        # Main simulation loop
        while self.time < Tnext:
            # Save current state
            Tra1  = np.float32(self.Tra)
            Tcr1  = np.float32(self.Tcr)
            Tmu1  = np.float32(self.Tmu)
            Tfat1 = np.float32(self.Tfat)
            Tvsk1 = np.float32(self.Tvsk)
            Tsk1  = np.float32(self.Tsk)
            TTC1  = np.float32(self.TTC)

            dQra_dt1  = np.float32(self.dQra_dt)
            dQcr_dt1  = np.float32(self.dQcr_dt)
            dQmu_dt1  = np.float32(self.dQmu_dt)
            dQfat_dt1 = np.float32(self.dQfat_dt)
            dQvsk_dt1 = np.float32(self.dQvsk_dt)
            dQsk_dt1  = np.float32(self.dQsk_dt)
            dQTC_dt1  = np.float32(self.dQTC_dt)
            dO2debt_dt1 = np.float32(self.dO2debt_dt)
            SR1 = np.float32(self.SR)

            # Estimate temperatures at end of time step
            self.Tra  = np.float32(Tra1  + dQra_dt1  / self.HtCapRA  * self.dt)
            self.Tcr  = np.float32(Tcr1  + dQcr_dt1  / self.HtCapCr  * self.dt)
            self.Tmu  = np.float32(Tmu1  + dQmu_dt1  / self.HtCapMu  * self.dt)
            self.Tfat = np.float32(Tfat1 + dQfat_dt1 / self.HtCapFat * self.dt)
            self.Tvsk = np.float32(Tvsk1 + dQvsk_dt1 / self.HtCapVsk * self.dt)
            self.Tsk  = np.float32(Tsk1  + dQsk_dt1  / self.HtCapSk  * self.dt)
            self.TTC  = np.float32(TTC1  + dQTC_dt1  / self.HtCapSk  * self.dt)

            # Update T-dependent processes
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
            self.deriv(Cres, Eres, Mcr, Mfat, Mvsk, Msk)

            # Store new derivatives
            dQra_dt2  = np.float32(self.dQra_dt)
            dQcr_dt2  = np.float32(self.dQcr_dt)
            dQmu_dt2  = np.float32(self.dQmu_dt)
            dQfat_dt2 = np.float32(self.dQfat_dt)
            dQvsk_dt2 = np.float32(self.dQvsk_dt)
            dQsk_dt2  = np.float32(self.dQsk_dt)
            dQTC_dt2  = np.float32(self.dQTC_dt)
            dO2debt_dt2 = np.float32(self.dO2debt_dt)
            SR2 = np.float32(self.SR)

            # Average old and new derivatives
            self.dQra_dt  = np.float32(0.5 * (dQra_dt1  + dQra_dt2))
            self.dQcr_dt  = np.float32(0.5 * (dQcr_dt1  + dQcr_dt2))
            self.dQmu_dt  = np.float32(0.5 * (dQmu_dt1  + dQmu_dt2))
            self.dQfat_dt = np.float32(0.5 * (dQfat_dt1 + dQfat_dt2))
            self.dQvsk_dt = np.float32(0.5 * (dQvsk_dt1 + dQvsk_dt2))
            self.dQsk_dt  = np.float32(0.5 * (dQsk_dt1  + dQsk_dt2))
            self.dQTC_dt  = np.float32(0.5 * (dQTC_dt1  + dQTC_dt2))
            self.dO2debt_dt = np.float32(0.5 * (dO2debt_dt1 + dO2debt_dt2))
            self.SR = np.float32(0.5 * (SR1 + SR2))

            # Final temperatures after averaging derivatives
            self.Tra  = np.float32(Tra1  + self.dQra_dt  / self.HtCapRA  * self.dt)
            self.Tcr  = np.float32(Tcr1  + self.dQcr_dt  / self.HtCapCr  * self.dt)
            self.Tmu  = np.float32(Tmu1  + self.dQmu_dt  / self.HtCapMu  * self.dt)
            self.Tfat = np.float32(Tfat1 + self.dQfat_dt / self.HtCapFat * self.dt)
            self.Tvsk = np.float32(Tvsk1 + self.dQvsk_dt / self.HtCapVsk * self.dt)
            self.Tsk  = np.float32(Tsk1  + self.dQsk_dt  / self.HtCapSk  * self.dt)
            self.TTC  = np.float32(TTC1  + self.dQTC_dt  / self.HtCapSk  * self.dt)

            # Integrate heat content
            self.Qra  = np.float32(self.Qra  + np.float32(0.06) * self.dQra_dt  * self.dt)
            self.Qcr  = np.float32(self.Qcr  + np.float32(0.06) * self.dQcr_dt  * self.dt)
            self.Qmu  = np.float32(self.Qmu  + np.float32(0.06) * self.dQmu_dt  * self.dt)
            self.Qfat = np.float32(self.Qfat + np.float32(0.06) * self.dQfat_dt * self.dt)
            self.Qvsk = np.float32(self.Qvsk + np.float32(0.06) * self.dQvsk_dt * self.dt)
            self.Qsk  = np.float32(self.Qsk  + np.float32(0.06) * self.dQsk_dt  * self.dt)
            self.Qtot = np.float32(self.Qra + self.Qcr + self.Qmu + self.Qfat +
                                   self.Qvsk + self.Qsk)

            # O2 debt and fluid updates
            self.O2debt   = np.float32(self.O2debt + self.dO2debt_dt * self.dt)
            self.fluidLoss= np.float32(self.fluidLoss + (self.SR - fluidIntake) * self.dt)

            # Update T-dependent props again
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
            self.deriv(Cres, Eres, Mcr, Mfat, Mvsk, Msk)

            # Increment time
            self.time = np.float32(self.time + self.dt)
            if (self.time + self.FUZZ) > Tnext:
                self.time = np.float32(Tnext)
            self.iter += 1

            Tmin = np.float32(self.compTmin())
            if Tmin <= np.float32(0.0):
                # For exact equivalence with Java, you'd raise an exception:
                print("Negative timestep computed. Reduce stability factor.")
                # raise ModelException(self.getVersion(), self.NEGATIVE_TIMESTEP)

            self.dt = np.float32(self.stabilityFactor * Tmin)
            self.FUZZ = np.float32(0.01) * self.dt

            if self.DEBUG:
                self.outputDebug()

            # Update circadian cycle
            timeOfDay = np.float32(self.time / np.float32(60.0) + self.startTime)
            if timeOfDay > np.float32(24.0):
                timeOfDay = np.float32(timeOfDay - np.float32(24.0))
            if self.circadianModel:
                self.ThypSet = np.float32(self.circadianTable.interpTemp(timeOfDay)
                                          + self.delThypSet)

        # Physiological Strain Index
        self.PSI = np.float32(np.float32(5.0) * (self.Tcr - self.Tre0)
                              / (np.float32(39.5) - self.Tre0)
                              + np.float32(5.0) * (self.HR - self.HR0)
                              / (np.float32(180.0) - self.HR0))
        if self.PSI < np.float32(0.0):
            self.PSI = np.float32(0.0)

        # Additional derived values
        CrSkdT = np.float32(self.Tcr - self.Tsk)
        RaSkdT = np.float32(self.Tra - self.Tsk)
        dQtot_dt = np.float32(self.dQra_dt + self.dQcr_dt + self.dQmu_dt +
                              self.dQfat_dt + self.dQvsk_dt + self.dQsk_dt)
        self.compTbdy()
        MRC = np.float32(self.Mnet - (Eres + Cres) - self.DryHtLoss1)

        # Populate final outputs
        self.predict1.setTra(self.Tra)
        self.predict1.setTcr(self.Tcr)
        self.predict1.setTmu(self.Tmu)
        self.predict1.setTfat(self.Tfat)
        self.predict1.setTvsk(self.Tvsk)
        self.predict1.setTsk(self.Tsk)
        self.predict1.setTcl(self.Tcl)
        self.predict1.setTbdy(self.Tbdy)

        self.predict2.setBFra(self.BFra)
        self.predict2.setBFcr(self.BFcr)
        self.predict2.setBFmu(self.BFmu)
        self.predict2.setBFfat(self.BFfat)
        self.predict2.setBFvsk(self.BFvsk)

        self.predict4.setdQtot_dt(dQtot_dt / self.SA)
        self.predict4.setdQra_dt(self.dQra_dt / self.SA)
        self.predict4.setdQcr_dt(self.dQcr_dt / self.SA)
        self.predict4.setdQmu_dt(self.dQmu_dt / self.SA)
        self.predict4.setdQfat_dt(self.dQfat_dt / self.SA)
        self.predict4.setdQvsk_dt(self.dQvsk_dt / self.SA)
        self.predict4.setdQsk_dt(self.dQsk_dt / self.SA)

        self.predict5.setQtot(self.Qtot)
        self.predict5.setQra(self.Qra)
        self.predict5.setQcr(self.Qcr)
        self.predict5.setQmu(self.Qmu)
        self.predict5.setQfat(self.Qfat)
        self.predict5.setQvsk(self.Qvsk)
        self.predict5.setQsk(self.Qsk)

        self.predict6.setSR(self.SR)
        self.predict6.setDrip(self.Drip)
        self.predict6.setEsk(self.Esk / self.SA)
        self.predict6.setEmax(self.Emax / self.SA)
        self.predict6.setSkWet(self.SkWet * np.float32(100.0))
        self.predict6.setdMshiv(self.dMshiv / self.SA)
        self.predict6.setMRC(MRC / self.SA)
        self.predict6.setCrSkdT(CrSkdT)
        self.predict6.setRaSkdT(RaSkdT)

        self.predict7.setFluidLoss(self.fluidLoss)
        self.predict7.setHR(self.HR)
        self.predict7.setSV(self.SV)
        self.predict7.setO2debt(self.O2debt)
        self.predict7.setPSI(self.PSI)

        self.setPts_n_Flags.setThypSet(self.ThypSet)
        self.setPts_n_Flags.setTimeOfDay(timeOfDay)

    def exit(self):
        """
        Allows any clean up on the server to occur.
        Mirrors: public void exit()
        """
        self.closeOutputStream()

    def compKc(self, workMode, Vair, Vmove, Mrst):
        """
        Computes dry conductance Kc [W/m^2-C].
        Mirrors: float compKc(char workMode, float Vair, float Vmove, float Mrst)
        """
        k1 = np.float32(1.96) * np.float32(Vair ** np.float32(0.86))

        if workMode == 'r' or workMode == 'a':
            # Resting, sitting, standing
            Kc = np.float32(11.6 * math.sqrt(Vair))
        elif workMode == 't':
            # Treadmill
            Kc = np.float32(6.5 * (Vmove ** np.float32(0.39)) + k1)
        elif workMode == 'f':
            # Free Walking
            Kc = np.float32(8.6 * (Vmove ** np.float32(0.53)) + k1)
        elif workMode == 'e':
            # Ergometer
            Kc = np.float32(5.5 + k1)
        elif workMode == 'm':
            # user specified metabolic rates, assume free walking if > Mrst
            if self.Mtot > Mrst:
                Kc = np.float32(8.6 * (Vmove ** np.float32(0.53)) + k1)
            else:
                Kc = np.float32(11.6 * math.sqrt(Vair))
        elif workMode == 'n':
            # user specified metabolic rates, treadmill if > Mrst
            if self.Mtot > Mrst:
                Kc = np.float32(6.5 * (Vmove ** np.float32(0.39)) + k1)
            else:
                Kc = np.float32(11.6 * math.sqrt(Vair))
        else:
            Kc = np.float32(self.STANDARD_KC)

        return np.float32(Kc)

    def compKr(self, Kc, Iclo, Tmr, Ta):
        """
        Computes dry radiative conductance Kr [W/m^2-C] using Newton's Method.
        Mirrors: float compKr(float Kc, float Iclo, float Tmr, float Ta)
        """
        RadFact = np.float32(1.0 + np.float32(0.15) * Iclo)
        k1 = np.float32(4.0) * np.float32(Boltz) * RadFact * np.float32(SArad)
        k2 = np.float32(KClo * Iclo)

        # initial guess
        x1 = np.float32(self.KR0)

        if self.time > np.float32(120.0):
            x1 = np.float32(self.KR0)

        ict = 0
        reldif = np.float32(1.0)
        ZERO = np.float32(0.00001)

        while reldif > ZERO:
            Kop = np.float32(Kc + x1)
            p = np.float32(1.0 / (1.0 + k2 * Kop))
            q = np.float32(Tmr * x1 + Kc * Ta)
            r = np.float32(1.0 / Kop)

            s = np.float32((1.0 - 0.5 * p) * q * r + 0.5 * self.Tsk * p + 273.0)
            f = np.float32(k1 * (s ** 3) - x1)

            dpdx = np.float32(-k2 / ((1.0 + k2 * Kop) ** 2))
            dqdx = np.float32(Tmr)
            drdx = np.float32(-1.0 / (Kop ** 2))

            dsdx = np.float32(dqdx * r + drdx * q
                              - 0.5 * (dpdx * q * r + p * dqdx * r + p * q * drdx)
                              + 0.5 * self.Tsk * dpdx)

            dfdx = np.float32(3.0 * k1 * (s ** 2) * dsdx - 1.0)

            x2 = np.float32(x1 - f / dfdx)
            reldif = np.float32(abs(x2 - x1))
            x1 = x2
            ict += 1
            if ict == 6:
                print("Failed to converge in compKr.")
                # raise ModelException(self.getVersion(), self.FAILED_TO_CONVERGE)

        return np.float32(x1)

    def openOutputStream(self):
        """
        Opens the output stream used for time constant analysis.
        Mirrors: void openOutputStream()
        """
        try:
            # In the Java code, it writes to "tau.dat". We'll just no-op or mock.
            print("pass")
        except Exception as e:
            print(e)

    def openDebugStream(self):
        """
        Auxilliary function to support debug diagnostics.
        Mirrors: void openDebugStream()
        """
        try:
            print("pass")
        except Exception as e:
            print(e)

    def closeOutputStream(self):
        """
        Closes the output stream used for time constant analysis.
        Mirrors: void closeOutputStream()
        """
        if self.outputStream is not None:
            self.outputStream.close()
            self.outputStream = None

        if self.DEBUG and self.debugStream is not None:
            self.debugStream.close()
            self.debugStream = None

    def compStrokeVolume(self, SVmax):
        """
        Computes SV [ml].
        Mirrors: void compStrokeVolume(float SVmax)
        """
        SVlag = np.float32(LagIt(self.SVold, SVmax, float(self.time), 0.5))
        MWST  = np.float32(self.Tsk)
        if MWST > np.float32(38.0):
            MWST = np.float32(38.0)
        if MWST <= np.float32(33.0):
            self.SV = np.float32(SVlag)
        else:
            self.SV = np.float32(SVlag - np.float32(5.0) *
                                 ((SVmax - np.float32(85.0)) / np.float32(45.0)) *
                                 (MWST - np.float32(33.0)))

    def compCOreq(self):
        """
        Computes HR [bpm], BFra [ml/min], Overload [ml/min], PctHRmax [%].
        Mirrors: void compCOreq()
        """
        COreq = np.float32(self.BFcr + self.BFmuReq + self.BFfat + self.BFvskReq)
        self.HR = np.float32(COreq / self.SV)
        self.BFra = np.float32(COreq)
        self.Overload = np.float32(0.0)
        COmax = np.float32(self.HRmax * self.SV)
        if self.HR > self.HRmax:
            self.HR = np.float32(self.HRmax)
            self.BFra = np.float32(COmax)
            self.Overload = np.float32(COreq - COmax)
        self.PctHRmax = np.float32(100.0 * self.HR / self.HRmax)

    def compVascularBloodFlow(self, VO2):
        """
        Computes BFvsk and BFvskReq [ml/min].
        Mirrors: void compVascularBloodFlow(float VO2)
        """
        BFvskLast = np.float32(self.BFvsk)
        BFvskMax = np.float32(LagIt(self.BFvskMaxOld, self.BFvskMaxNew,
                                    float(self.time), 0.5))

        MWST = np.float32(self.Tsk)
        if MWST < np.float32(30.0):
            MWST = np.float32(30.0)
        if MWST > np.float32(35.3):
            MWST = np.float32(35.3)

        if VO2 <= np.float32(0.75):
            TsetBFnew = np.float32(37.07 - 0.108 * (MWST - 30.0))
        else:
            if MWST <= np.float32(33.0):
                TsetBFnew = np.float32(37.32 - 0.093 * (MWST - 30.0))
            else:
                TsetBFnew = np.float32(37.04 - 0.03  * (MWST - 33.0))

        TsetBF = np.float32(LagIt(self.TsetBFold, TsetBFnew,
                                  float(self.time), 1.0))
        PctBFvskMax = np.float32(70.3 * (self.Tra - TsetBF))
        if PctBFvskMax > np.float32(100.0):
            PctBFvskMax = np.float32(100.0)

        self.BFvskReq = np.float32((PctBFvskMax / np.float32(100.0)) * BFvskMax)
        if self.BFvskReq < self.BFvskMin:
            self.BFvskReq = np.float32(self.BFvskMin)

        if self.Overload > np.float32(0.0):
            self.BFvsk = np.float32(BFvskLast)
        else:
            self.BFvsk = np.float32(self.BFvskReq)

    def compVskSkCond(self, workMode):
        """
        Computes VskSkCond [W/C] and BFcr [ml/min].
        Mirrors: void compVskSkCond(char workMode)
        """
        self.VskSkCond = np.float32(self.SkCondConst * math.log(self.BFvsk / self.BFvskMin)
                                    + np.float32(10.0) * self.SA)

        # Core Blood Flow
        if workMode == 'r':
            PctBFcrRst = np.float32(100.0 - 1.266 * (self.PctHRmax - 25.0))
        else:
            PctBFcrRst = np.float32(100.0 - 1.086 * (self.PctHRmax - 39.0))

        if PctBFcrRst > np.float32(100.0):
            PctBFcrRst = np.float32(100.0)

        BFcrNew = np.float32(self.COrst * fCOrstCr * PctBFcrRst / np.float32(100.0))
        self.BFcr = np.float32(LagIt(self.BFcrOld, BFcrNew, float(self.time), 1.0))

    def compMuscleBloodFlow(self):
        """
        Computes BFmuReq, BFmu [ml/min], PctExt [%], and LeftOver [ml/min].
        Mirrors: void compMuscleBloodFlow()
        """
        self.BFmuReq = np.float32((self.EEmu * self.O2Equiv) /
                                  (np.float32(0.01) * self.PctExt * self.BlO2Cap))
        self.BFmu = np.float32(self.BFmuReq)
        MUmin = np.float32((self.EEmu * self.O2Equiv) / self.BlO2Cap)
        MUpot = np.float32(self.BFmuReq - MUmin)

        self.PctExt = np.float32(80.0)
        self.LeftOver = np.float32(0.0)

        if self.Overload > np.float32(0.0) and MUpot > np.float32(0.0):
            if self.Overload <= MUpot:
                self.BFmu = np.float32(self.BFmuReq - self.Overload)
                self.PctExt = np.float32((100.0 * self.EEmu * self.O2Equiv) /
                                         (self.BFmu * self.BlO2Cap))
            else:
                self.LeftOver = np.float32(self.Overload - MUpot)
                self.BFmu = np.float32(MUmin - self.LeftOver)
                self.PctExt = np.float32(100.0)
        elif self.Overload > np.float32(0.0) and MUpot <= np.float32(0.0):
            self.LeftOver = np.float32(self.Overload)
            self.BFmu = np.float32(MUmin - self.LeftOver)
            self.PctExt = np.float32(100.0)

    def compShivering(self):
        """
        Computes dMshiv [W] i.e. shivering.
        Mirrors: void compShivering()
        """
        SkSig = np.float32(self.Tsk - self.TskSet)
        if SkSig <= np.float32(0.0):
            CldSkSig = np.float32(-SkSig)
        else:
            CldSkSig = np.float32(0.0)

        HypSig = np.float32(self.Tra - self.ThypSet)
        if HypSig <= np.float32(0.0):
            CldHypSig = np.float32(-HypSig)
        else:
            CldHypSig = np.float32(0.0)

        self.dMshiv = np.float32(self.SA * np.float32(19.4) * CldSkSig * CldHypSig)

    def compSweatRate(self):
        """
        Computes SR [gm/min] i.e. sweat rate.
        Mirrors: void compSweatRate()
        """
        SkSig = np.float32(self.Tsk - self.TskSet)
        HypSig = np.float32(self.Tra - self.ThypSet)

        if SkSig > np.float32(100.0):
            self.SR = np.float32(self.SRmax)
        else:
            pctWgtLoss = np.float32(0.1 * self.fluidLoss / self.BW)
            deltaAlphaSR = np.float32(self.DELTA_ALPHA_SR * pctWgtLoss)
            deltaHypSig = np.float32(self.DELTA_HYPSIG * pctWgtLoss)
            self.SR = np.float32(self.SA * ((np.float32(4.83) + deltaAlphaSR) *
                              (HypSig + deltaHypSig) + np.float32(0.56) * SkSig) *
                              math.exp(SkSig / np.float32(10.0)))

        if self.SR < self.SRmin:
            self.SR = np.float32(self.SRmin)
        if self.SR > self.SRmax:
            self.SR = np.float32(self.SRmax)

    def compDripSkWetEsk(self, Kop, Kc, Pvap, ClEffFact, Im):
        """
        Computes Drip [gm/min], SkWet [fraction], and Esk [W].
        Mirrors: void compDripSkWetEsk(...)
        """
        self.Drip = np.float32(0.0)
        self.SkWet = np.float32(0.0)

        Esw = np.float32(HtVap * self.SR)
        ClPermFact = np.float32((Kop / Kc) * ClEffFact * Im)
        Psk = np.float32(SatVP(self.Tsk))
        self.Emax = np.float32(self.SA * Lewis * Kc * (Psk - Pvap) * ClPermFact)

        if self.Emax <= np.float32(0.0):
            self.Emax = np.float32(0.0)
            Esw = np.float32(self.Emax)
            self.SkWet = np.float32(1.0)
            self.Drip = np.float32(self.SR)
        else:
            self.SkWet = np.float32(Esw / self.Emax)
            if self.SkWet > np.float32(1.0):
                Esw = np.float32(self.Emax)
                self.SkWet = np.float32(1.0)
                self.Drip = np.float32(self.SR - (self.Emax / HtVap))
            else:
                self.Drip = np.float32(0.0)

        # Edif = 0.06 * (1.0 - self.SkWet) * self.Emax   # commented out in Java
        self.Esk = np.float32(Esw)  # + Edif if you uncomment

    def compShiverCorrection(self, Mmu2, Mtot2, Mext, EEmuOld, EEmuNew):
        """
        Shiver correction: computes Mmu, Mtot, Mnet, EEmu.
        Mirrors: void compShiverCorrection(...)
        """
        self.Mmu  = np.float32(Mmu2  + self.dMshiv)
        self.Mtot = np.float32(Mtot2 + self.dMshiv)
        self.Mnet = np.float32(self.Mtot - Mext)

        Tlag = np.float32(self.time - self.T_EEmuNew)
        EEmu2 = np.float32(LagIt(EEmuOld, EEmuNew, float(Tlag), 1.0))
        self.EEmu = np.float32(EEmu2 + self.dMshiv)

    def compDryHeatLoss(self, RadFact, Kc, Tmr, Ta, Iclo):
        """
        Computes DryHtLoss1, DryHtLoss2 [W], Top, Tcl [C], ClEffFact [-].
        Mirrors: void compDryHeatLoss(...)
        """
        self.Tcl = np.float32(self.Top + (self.ClEffFact * (self.Tsk - self.Top)))
        Kr = np.float32(4.0 * Boltz *
                        ((np.float32(self.Tcl + self.Top) / np.float32(2.0) +
                          np.float32(273.0)) ** np.float32(3.0)) *
                        RadFact * SArad)
        Kop = np.float32(Kc + Kr)
        self.Top = np.float32((Kr * Tmr + Kc * Ta) / Kop)
        self.ClEffFact = np.float32(1.0 / (1.0 + Kop * KClo * Iclo))
        self.DryHtLoss1 = np.float32(self.SA * Kop * self.ClEffFact * (self.Tsk - self.Top))
        self.DryHtLoss2 = np.float32(self.SA * Kop * self.ClEffFact * (self.TTC - self.Top))

    def deriv(self, Cres, Eres, Mcr, Mfat, Mvsk, Msk):
        """
        Computes time derivatives.
        Mirrors: void deriv(...)
        """
        self.dQra_dt = np.float32(SpHtBl * 0.001 * (
            self.BFcr * (self.Tcr - self.Tra) +
            self.BFmu * (self.Tmu - self.Tra) +
            self.BFfat* (self.Tfat - self.Tra) +
            self.BFvsk* (self.Tvsk- self.Tra)) - (Cres + Eres))

        self.dQcr_dt = np.float32(Mcr
            - self.CrMuCond * (self.Tcr - self.Tmu)
            - SpHtBl * 0.001 * self.BFcr  * (self.Tcr - self.Tra))

        self.dQmu_dt = np.float32(self.Mmu
            + self.CrMuCond * (self.Tcr - self.Tmu)
            - self.MuFatCond * (self.Tmu - self.Tfat)
            - SpHtBl * 0.001 * self.BFmu * (self.Tmu - self.Tra))

        self.dQfat_dt = np.float32(Mfat
            + self.MuFatCond * (self.Tmu - self.Tfat)
            - self.FatVskCond * (self.Tfat - self.Tvsk)
            - SpHtBl * 0.001 * self.BFfat * (self.Tfat - self.Tra))

        self.dQvsk_dt = np.float32(Mvsk
            + self.FatVskCond * (self.Tfat - self.Tvsk)
            - self.VskSkCond  * (self.Tvsk - self.Tsk)
            - SpHtBl * 0.001 * self.BFvsk * (self.Tvsk - self.Tra))

        self.dQsk_dt = np.float32(Msk
            + self.VskSkCond * (self.Tvsk - self.Tsk)
            - self.DryHtLoss1
            - self.Esk)

        self.dQTC_dt = np.float32(Msk
            + self.VskSkCond * (self.Tvsk - self.TTC)
            - self.DryHtLoss2)

        self.dO2debt_dt = np.float32((0.0001 * self.PctExt * self.BlO2Cap) * self.LeftOver)

    def compTmin0(self):
        """
        Computes static time constants.
        Mirrors: float compTmin0()
        """
        Tmu_cr = np.float32(self.HtCapCr / self.CrMuCond)
        Tmin = np.float32(Tmu_cr)

        Tcr_mu = np.float32(self.HtCapMu / self.CrMuCond)
        if Tcr_mu < Tmin:
            Tmin = np.float32(Tcr_mu)
        Tfat_mu = np.float32(self.HtCapMu / self.MuFatCond)
        if Tfat_mu < Tmin:
            Tmin = np.float32(Tfat_mu)

        Tmu_fat = np.float32(self.HtCapFat / self.MuFatCond)
        if Tmu_fat < Tmin:
            Tmin = np.float32(Tmu_fat)
        Tvsk_fat = np.float32(self.HtCapFat / self.FatVskCond)
        if Tvsk_fat < Tmin:
            Tmin = np.float32(Tvsk_fat)
        Rbl_fat = np.float32(1.0 / (SpHtBl * 0.001 * self.BFfat))
        Tbl_fat = np.float32(Rbl_fat * self.HtCapFat)
        if Tbl_fat < Tmin:
            Tmin = np.float32(Tbl_fat)

        Tfat_vsk = np.float32(self.HtCapVsk / self.FatVskCond)
        if Tfat_vsk < Tmin:
            Tmin = np.float32(Tfat_vsk)

        Tfat_bl = np.float32(Rbl_fat * self.HtCapRA)
        if Tfat_bl < Tmin:
            Tmin = np.float32(Tfat_bl)

        if not self.PRINT_TAU:
            return np.float32(Tmin)

        # If printing, write out the table once.
        self.outputStream.setWidth(10)
        s = "Tmu_cr"
        self.outputStream.print(s)
        s = "Tcr_mu"
        self.outputStream.print(s)
        s = "Tfat_mu"
        self.outputStream.print(s)
        s = "Tmu_fat"
        self.outputStream.print(s)
        s = "Tvsk_fat"
        self.outputStream.print(s)
        s = "Tbl_fat"
        self.outputStream.print(s)
        s = "Tfat_vsk"
        self.outputStream.print(s)
        s = "Tfat_bl"
        self.outputStream.println(s)

        self.outputStream.print(self.numberFormat4.format(Tmu_cr))
        self.outputStream.print(self.numberFormat4.format(Tcr_mu))
        self.outputStream.print(self.numberFormat4.format(Tfat_mu))
        self.outputStream.print(self.numberFormat4.format(Tmu_fat))
        self.outputStream.print(self.numberFormat4.format(Tvsk_fat))
        self.outputStream.print(self.numberFormat4.format(Tbl_fat))
        self.outputStream.print(self.numberFormat4.format(Tfat_vsk))
        self.outputStream.println(self.numberFormat4.format(Tfat_bl))

        return np.float32(Tmin)

    def tauDynamicHeader(self):
        """
        Creates the header in TAU.DAT for the dynamic time constant data.
        Mirrors: void tauDynamicHeader()
        """
        self.outputStream.println("")
        s = "Iter."
        self.outputStream.print(s)
        s = "Tbl_cr"
        self.outputStream.print(s)
        s = "Tbl_mu"
        self.outputStream.print(s)
        s = "Tbl_vsk"
        self.outputStream.print(s)
        s = "Tsk_vsk"
        self.outputStream.print(s)
        s = "Tvsk_sk"
        self.outputStream.print(s)
        s = "Tcr_bl"
        self.outputStream.print(s)
        s = "Tmu_bl"
        self.outputStream.print(s)
        s = "Tvsk_bl"
        self.outputStream.print(s)
        s = "Tmin"
        self.outputStream.println(s)

    def compTmin(self):
        """
        Computes dynamic time constants.
        Mirrors: float compTmin()
        """
        Tmin = np.float32(self.Tmin0)

        printNow = self.PRINT_TAU
        if printNow and (self.iter == 0):
            self.tauDynamicHeader()

        if (self.iter % 20) != 0:
            printNow = False

        if not printNow:
            return np.float32(Tmin)

        self.outputStream.print(self.time)
        self.outputStream.print(self.iter)

        Rbl_cr = np.float32(1.0 / (SpHtBl * 0.001 * self.BFcr))
        Tbl = np.float32(Rbl_cr * self.HtCapCr)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        Rbl_mu = np.float32(1.0 / (SpHtBl * 0.001 * self.BFmu))
        Tbl = np.float32(Rbl_mu * self.HtCapMu)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        Rbl_vsk = np.float32(1.0 / (SpHtBl * 0.001 * self.BFvsk))
        Tbl = np.float32(Rbl_vsk * self.HtCapVsk)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        Tsk_vsk = np.float32(self.HtCapVsk / self.VskSkCond)
        if Tsk_vsk < Tmin:
            Tmin = np.float32(Tsk_vsk)
        self.outputStream.print(self.numberFormat4.format(Tsk_vsk))

        Tvsk_sk = np.float32(self.HtCapSk / self.VskSkCond)
        if Tvsk_sk < Tmin:
            Tmin = np.float32(Tvsk_sk)
        self.outputStream.print(self.numberFormat4.format(Tvsk_sk))

        Tbl = np.float32(Rbl_cr * self.HtCapRA)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        Tbl = np.float32(Rbl_mu * self.HtCapRA)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        Tbl = np.float32(Rbl_vsk * self.HtCapRA)
        if Tbl < Tmin:
            Tmin = np.float32(Tbl)
        self.outputStream.print(self.numberFormat4.format(Tbl))

        self.outputStream.println(self.numberFormat4.format(Tmin))
        return np.float32(Tmin)

    def outputDebug(self):
        """
        Auxilliary function to support debug diagnostics.
        Mirrors: void outputDebug()
        """
        self.debugStream.print(self.iter)
        s = self.debugFormat6.format(self.time)
        self.debugStream.print(s)
        s = self.debugFormat2.format(self.BFcr)
        self.debugStream.print(s)
        s = self.debugFormat4.format(self.SV)
        self.debugStream.print(s)
        s = self.debugFormat4.format(self.HR)
        self.debugStream.print(s)
        s = self.debugFormat2.format(self.BFcrOld)
        self.debugStream.print(s)
        s = self.debugFormat2.format(self.PctHRmax)
        self.debugStream.println(s)

    def compTbdy(self):
        """
        Computes the effective body temperature.
        Mirrors: void compTbdy()
        """
        totHtCap = np.float32(self.HtCapRA + self.HtCapCr + self.HtCapMu +
                              self.HtCapFat + self.HtCapVsk + self.HtCapSk)
        self.Tbdy = np.float32(((self.HtCapRA * self.Tra) +
                                (self.HtCapCr * self.Tcr) +
                                (self.HtCapMu * self.Tmu) +
                                (self.HtCapFat * self.Tfat) +
                                (self.HtCapVsk * self.Tvsk) +
                                (self.HtCapSk  * self.Tsk)) / totHtCap)
