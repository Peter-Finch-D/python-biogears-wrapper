"""
tissue_conductance.py

A Python module replicating the TissueConductance class from TissueConductance.java.
"""

from scenario_constants import (
    UNDEFINED_FLOAT, PI,
    fBWnfatCr, fBWnfatMu, fBWnfatVsk, fBWnfatSk, fBWnfatRa,
    Dfat, Dnfat, kcr, kmu, kfat, kvsk, ksk
)
from scenario_constants import SpHtBl    # If needed
from scenario_constants import NO_ACCLIMATION, PART_ACCLIMATION, FULL_ACCLIMATION
from scenario_constants import DEHYD_NORMAL, DEHYD_MODERATE, DEHYD_SEVERE

class TissueConductance:
    """
    Python equivalent of the TissueConductance Java class.
    """
    def __init__(self):
        self.CrMuCond   = UNDEFINED_FLOAT
        self.MuFatCond  = UNDEFINED_FLOAT
        self.FatVskCond = UNDEFINED_FLOAT
        self.VskSkCond  = UNDEFINED_FLOAT
        self.SkCondConst= UNDEFINED_FLOAT

    def copyOf(self, other: 'TissueConductance'):
        self.CrMuCond   = other.CrMuCond
        self.MuFatCond  = other.MuFatCond
        self.FatVskCond = other.FatVskCond
        self.VskSkCond  = other.VskSkCond
        self.SkCondConst= other.SkCondConst

    def getCrMuCond(self):
        return self.CrMuCond

    def getMuFatCond(self):
        return self.MuFatCond

    def getFatVskCond(self):
        return self.FatVskCond

    def getVskSkCond(self):
        return self.VskSkCond

    def getSkCondConst(self):
        return self.SkCondConst

    def setCrMuCond(self, val: float):
        self.CrMuCond = val

    def setMuFatCond(self, val: float):
        self.MuFatCond = val

    def setFatVskCond(self, val: float):
        self.FatVskCond = val

    def setVskSkCond(self, val: float):
        self.VskSkCond = val

    def setSkCondConst(self, val: float):
        self.SkCondConst = val

    def compute(self, inputs1):
        """
        In Java, inputs1 is an instance of Inputs1, which has:
          - getPctFat(), getBW(), getSA(), ...
        We'll assume here that inputs1 is a Python object
        with attributes: .PctFat, .BW, .SA
        """
        PctFat = inputs1.PctFat
        BW     = inputs1.BW
        SA     = inputs1.SA

        # replicate TissueConductance.java logic
        # 1) Wfat/Wnfat in grams
        Wfat  = 10.0 * PctFat * BW      # g
        Wnfat = 10.0 * (100.0 - PctFat) * BW

        # Minnesota eqn for body density:
        Dbdy = 457.0 / (PctFat + 414.2)
        Vbdy = 1000.0 * BW / Dbdy       # cm^3
        SAbdy= SA * 10000.0             # cm^2
        Lcyl = (SAbdy**2) / (4.0 * PI * Vbdy)
        Rcyl = 2.0 * Vbdy / SAbdy

        # volumes of compartments
        Vcr  = (fBWnfatCr  * Wnfat) / Dnfat
        Vmu  = (fBWnfatMu  * Wnfat) / Dnfat
        Vfat = (Wfat)              / Dfat
        Vvsk = (fBWnfatVsk * Wnfat) / Dnfat
        Vsk  = (fBWnfatSk  * Wnfat) / Dnfat
        Vra  = (fBWnfatRa  * Wnfat) / Dnfat

        # Radii calculations (adapted from Java)
        VOUTsk = Vbdy
        VINsk  = VOUTsk - Vsk
        ROUTsk = Rcyl
        RINsk  = (VINsk / (PI * Lcyl))**0.5
        THsk   = ROUTsk - RINsk

        VOUTvsk = VINsk
        VINvsk  = VOUTvsk - Vvsk
        ROUTvsk = RINsk
        RINvsk  = (VINvsk / (PI * Lcyl))**0.5
        THvsk   = ROUTvsk - RINvsk

        VOUTfat = VINvsk
        VINfat  = VOUTfat - Vfat
        ROUTfat = RINvsk
        RINfat  = (VINfat / (PI * Lcyl))**0.5
        THfat   = ROUTfat - RINfat

        VOUTmu = VINfat
        VINmu  = VOUTmu - Vmu
        ROUTmu = RINfat
        RINmu  = (VINmu / (PI * Lcyl))**0.5
        THmu   = ROUTmu - RINmu

        VOUTcr = VINmu
        VINcr  = 0.0
        ROUTcr = RINmu
        RINcr  = 0.0
        THcr   = ROUTcr - RINcr

        # half-volumes
        HVsk  = Vsk / 2.0
        HVvsk = Vvsk / 2.0
        HVfat = Vfat / 2.0
        HVmu  = Vmu / 2.0
        HVcr  = Vcr / 2.0

        VMCsk  = VOUTsk - HVsk
        VMCvsk = VOUTvsk - HVvsk
        VMCfat = VOUTfat - HVfat
        VMCmu  = VOUTmu - HVmu
        VMCcr  = VOUTcr - HVcr

        RMCsk  = (VMCsk  / (PI * Lcyl))**0.5
        RMCvsk = (VMCvsk / (PI * Lcyl))**0.5
        RMCfat = (VMCfat / (PI * Lcyl))**0.5
        RMCmu  = (VMCmu  / (PI * Lcyl))**0.5
        RMCcr  = (VMCcr  / (PI * Lcyl))**0.5

        RMPcrmu   = (RMCcr  + RMCmu ) / 2.0
        RMPmufat  = (RMCmu  + RMCfat) / 2.0
        RMPfatvsk = (RMCfat + RMCvsk) / 2.0
        RMPvsksk  = (RMCvsk + RMCsk ) / 2.0

        AMPcrmu   = (RMPcrmu)   * 2.0 * PI * Lcyl
        AMPmufat  = (RMPmufat)  * 2.0 * PI * Lcyl
        AMPfatvsk = (RMPfatvsk) * 2.0 * PI * Lcyl
        AMPvsksk  = (RMPvsksk)  * 2.0 * PI * Lcyl

        Rcrmu   = (RMCmu  - RMCcr ) / 2.0
        Rmufat  = (RMCfat - RMCmu ) / 2.0
        Rfatvsk = (RMCvsk - RMCfat) / 2.0
        Rvsksk  = (RMCsk  - RMCvsk) / 2.0

        # function to do the Ka/Kb => Kave step
        def k_combine(kA, kB, delta_r):
            if delta_r <= 0.0:
                return 0.0
            Ka = kA / delta_r
            Kb = kB / delta_r
            tmp = (1.0 / Ka) + (1.0 / Kb)
            return 1.0 / tmp

        # compute each conduction:
        Kcrmu   = AMPcrmu   * 0.0001 * k_combine(kcr, kmu, Rcrmu)
        Kmufat  = AMPmufat  * 0.0001 * k_combine(kmu, kfat, Rmufat)
        Kfatvsk = AMPfatvsk * 0.0001 * k_combine(kfat, kvsk, Rfatvsk)
        Kvsksk  = AMPvsksk  * 0.0001 * k_combine(kvsk, ksk, Rvsksk)

        # in original code, they set VskSkCond to 55.233f * SA as "startup value"
        # then set SkCondConst = (Kvsksk - (10 * SA))/2.414f. 
        # We replicate that logic here:
        # setCrMuCond(16.855 * SA), etc. is from comments, but let's do the exact approach:
        self.setCrMuCond(Kcrmu)
        self.setMuFatCond(Kmufat)
        self.setFatVskCond(Kfatvsk)
        # We follow the original code:
        # setVskSkCond(55.233f * SA) at thermoneutral start
        # then final line: setSkCondConst( (Kvsksk - (10 * SA)) / 2.414f )
        self.setVskSkCond(55.233 * SA)

        self.setSkCondConst( (Kvsksk - (10.0 * SA)) / 2.414 )

