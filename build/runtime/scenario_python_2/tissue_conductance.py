# =============================================================================
# tissue_conductance.py
#
# This Python module is a direct translation of the Java class "TissueConductance"
# from the provided code. All functionality is intended to mirror the original
# Java version as closely as possible, line-by-line, preserving method names,
# logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" (which provides constants
#     like UNDEFINED_FLOAT, PI, fBWnfatCr, etc.) and a class "Inputs1" that
#     provides getPctFat(), getBW(), getSA() and so on.
#   - The usage example at the bottom (showing how to import this file into
#     scenario_model.py) is abridged only. Everything else is full and
#     unabridged.
# =============================================================================

# We assume scenario_constants.py is in the same package (scenario_python_2).
# If it's in a different package structure, adjust the import accordingly.
from scenario_constants import (
    UNDEFINED_FLOAT, PI, Dnfat, Dfat,
    fBWnfatCr, fBWnfatMu, fBWnfatVsk, fBWnfatSk, fBWnfatRa,  # watch out for actual naming
    kcr, kmu, kfat, kvsk, ksk,
)

# However, from the Java code, we see these constants are used in compute():
#   fBWnfatCr, fBWnfatMu, fBWnfatVsk, fBWnfatSk, fBWnfatRa,
#   kcr, kmu, kfat, kvsk, ksk
#   plus references like "PI", "Dfat", "Dnfat".
#
# Also note references to `457 / (PctFat + 414.2f)` for body density. That is
# just a literal usage in the code. We do not store it in the constants.
#
# We'll define a TissueConductance class that has the same structure as Java.

class TissueConductance:
    """
    A container for tissue conductance parameters. Translated from:
    mil.army.usariem.scenario.server.data.TissueConductance
    """

    def __init__(self):
        """
        Constructs with no argument. Mirrors the Java constructor.
        """
        self.CrMuCond = UNDEFINED_FLOAT
        self.MuFatCond = UNDEFINED_FLOAT
        self.FatVskCond = UNDEFINED_FLOAT
        self.VskSkCond = UNDEFINED_FLOAT
        self.SkCondConst = UNDEFINED_FLOAT

    def copyOf(self, tissueConductance):
        """
        Makes a copy of the specified object.
        Mirrors: public void copyOf(TissueConductance tissueConductance)
        """
        self.CrMuCond = tissueConductance.CrMuCond
        self.MuFatCond = tissueConductance.MuFatCond
        self.FatVskCond = tissueConductance.FatVskCond
        self.VskSkCond = tissueConductance.VskSkCond
        self.SkCondConst = tissueConductance.SkCondConst

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getCrMuCond(self):
        """Returns the CrMuCond value."""
        return self.CrMuCond

    def getMuFatCond(self):
        """Returns the MuFatCond value."""
        return self.MuFatCond

    def getFatVskCond(self):
        """Returns the FatVskCond value."""
        return self.FatVskCond

    def getVskSkCond(self):
        """Returns the VskSkCond value."""
        return self.VskSkCond

    def getSkCondConst(self):
        """Returns the SkCondConst value."""
        return self.SkCondConst

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setCrMuCond(self, val):
        """Sets the CrMuCond value."""
        self.CrMuCond = val

    def setMuFatCond(self, val):
        """Sets the MuFatCond value."""
        self.MuFatCond = val

    def setFatVskCond(self, val):
        """Sets the FatVskCond value."""
        self.FatVskCond = val

    def setVskSkCond(self, val):
        """Sets the VskSkCond value."""
        self.VskSkCond = val

    def setSkCondConst(self, val):
        """Sets the SkCondConst value."""
        self.SkCondConst = val

    # -------------------------------------------------------------------------
    # compute()
    # -------------------------------------------------------------------------
    def compute(self, inputs1):
        """
        Computes conductances based on Inputs1 data.
        Mirrors: public void compute(Inputs1 inputs1)
        """
        # from inputs1 we need getPctFat(), getBW(), getSA().
        PctFat = inputs1.getPctFat()
        BW = inputs1.getBW()
        SA = inputs1.getSA()
        pi = float(PI)

        # Calc weight of fat and nonfat
        Wfat  = 10.0 * PctFat * BW          # grams
        Wnfat = 10.0 * (100.0 - PctFat) * BW  # grams

        # Calc body density (Dbdy)
        Dbdy = 457.0 / (PctFat + 414.2)  # Minnesota equation

        # Calc body volume (Vbdy) in cm^3
        Vbdy = 1000.0 * BW / Dbdy

        # Calculate body surface area (SAbdy) in cm^2
        SAbdy = SA * 10000.0

        # Cylinder length Lcyl (mirroring the Java code)
        # Lcyl = (SAbdy^2) / (4 * pi * Vbdy)
        Lcyl = ((SAbdy ** 2) / (4.0 * pi * Vbdy))

        # Cylinder radius Rcyl = 2 * Vbdy / SAbdy
        Rcyl = 2.0 * Vbdy / SAbdy

        # Calculate compartment volumes
        # references constants fBWnfatCr, fBWnfatMu, etc. from scenario_constants
        Vcr  = (fBWnfatCr  * Wnfat) / float(Dnfat)
        Vmu  = (fBWnfatMu  * Wnfat) / float(Dnfat)
        Vfat = (Wfat)               / float(Dfat)
        Vvsk = (fBWnfatVsk * Wnfat) / float(Dnfat)
        Vsk  = (fBWnfatSk  * Wnfat) / float(Dnfat)
        Vra  = (fBWnfatRa  * Wnfat) / float(Dnfat)  # not used further

        # The calculations for the radius at compartments
        VOUTsk = Vbdy
        VINsk  = VOUTsk - Vsk
        ROUTsk = Rcyl
        RINsk  = (VINsk / (pi * Lcyl)) ** 0.5
        THsk   = ROUTsk - RINsk

        VOUTvsk = VINsk
        VINvsk  = VOUTvsk - Vvsk
        ROUTvsk = RINsk
        RINvsk  = (VINvsk / (pi * Lcyl)) ** 0.5
        THvsk   = ROUTvsk - RINvsk

        VOUTfat = VINvsk
        VINfat  = VOUTfat - Vfat
        ROUTfat = RINvsk
        RINfat  = (VINfat / (pi * Lcyl)) ** 0.5
        THfat   = ROUTfat - RINfat

        VOUTmu = VINfat
        VINmu  = VOUTmu - Vmu
        ROUTmu = RINfat
        RINmu  = (VINmu / (pi * Lcyl)) ** 0.5
        THmu   = ROUTmu - RINmu

        VOUTcr = VINmu
        VINcr  = 0
        ROUTcr = RINmu
        RINcr  = 0
        THcr   = ROUTcr - RINcr

        # find half-volumes
        HVsk  = Vsk  / 2.0
        HVvsk = Vvsk / 2.0
        HVfat = Vfat / 2.0
        HVmu  = Vmu  / 2.0
        HVcr  = Vcr  / 2.0

        # volume at center of mass
        VMCsk  = VOUTsk - HVsk
        VMCvsk = VOUTvsk - HVvsk
        VMCfat = VOUTfat - HVfat
        VMCmu  = VOUTmu  - HVmu
        VMCcr  = VOUTcr  - HVcr

        # radii at mass centers
        RMCsk  = (VMCsk  / (pi * Lcyl)) ** 0.5
        RMCvsk = (VMCvsk / (pi * Lcyl)) ** 0.5
        RMCfat = (VMCfat / (pi * Lcyl)) ** 0.5
        RMCmu  = (VMCmu  / (pi * Lcyl)) ** 0.5
        RMCcr  = (VMCcr  / (pi * Lcyl)) ** 0.5

        # midpoint radii
        RMPcrmu   = (RMCcr + RMCmu)   / 2.0
        RMPmufat  = (RMCmu + RMCfat)  / 2.0
        RMPfatvsk = (RMCfat + RMCvsk) / 2.0
        RMPvsksk  = (RMCvsk + RMCsk)  / 2.0

        # midpoint cylinder areas
        AMPcrmu   = (RMPcrmu)   * 2.0 * pi * Lcyl
        AMPmufat  = (RMPmufat)  * 2.0 * pi * Lcyl
        AMPfatvsk = (RMPfatvsk) * 2.0 * pi * Lcyl
        AMPvsksk  = (RMPvsksk)  * 2.0 * pi * Lcyl

        # (delta r)/2
        Rcrmu   = (RMCmu  - RMCcr)  / 2.0
        Rmufat  = (RMCfat - RMCmu)  / 2.0
        Rfatvsk = (RMCvsk - RMCfat) / 2.0
        Rvsksk  = (RMCsk  - RMCvsk) / 2.0

        # Ka, Kb, Kave
        # for CrMu:
        Ka   = kcr / Rcrmu
        Kb   = kmu / Rcrmu
        Kave = (1.0 / Ka) + (1.0 / Kb)
        Kave = 1.0 / Kave
        Kcrmu = AMPcrmu * 0.0001 * Kave

        # MuFat
        Ka   = kmu / Rmufat
        Kb   = kfat / Rmufat
        Kave = (1.0 / Ka) + (1.0 / Kb)
        Kave = 1.0 / Kave
        Kmufat = AMPmufat * 0.0001 * Kave

        # FatVsk
        Ka   = kfat / Rfatvsk
        Kb   = kvsk / Rfatvsk
        Kave = (1.0 / Ka) + (1.0 / Kb)
        Kave = 1.0 / Kave
        Kfatvsk = AMPfatvsk * 0.0001 * Kave

        # VskSk
        Ka   = kvsk / Rvsksk
        Kb   = ksk  / Rvsksk
        Kave = (1.0 / Ka) + (1.0 / Kb)
        Kave = 1.0 / Kave
        Kvsksk = AMPvsksk * 0.0001 * Kave

        # set
        self.setCrMuCond(Kcrmu)
        self.setMuFatCond(Kmufat)
        self.setFatVskCond(Kfatvsk)
        # 55.233f * SA is from the code's example, but the code does:
        #   setVskSkCond( 55.233f * SA ); // "Used as the startup value..."
        # We replicate that exactly, though note the Java code actually
        # calculates Kvsksk but doesn't use it for setVskSkCond?
        # The code sets: setVskSkCond( 55.233f * SA );
        # So we do that here:
        self.setVskSkCond(55.233 * SA)

        # SkCondConst is (Kvsksk - (10 * SA)) / 2.414f
        # from the code, we replicate that:
        self.setSkCondConst((Kvsksk - (10.0 * SA)) / 2.414)


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.tissue_conductance import TissueConductance
#   from scenario_python_2.scenario_constants import UNDEFINED_FLOAT
#   # then create and use it:
#   tc = TissueConductance()
#   tc.compute(inputs1)
#   print(tc.getCrMuCond())
#
# =============================================================================
