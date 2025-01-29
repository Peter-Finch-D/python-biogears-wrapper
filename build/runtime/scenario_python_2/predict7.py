# =============================================================================
# predict7.py
#
# This Python module is a direct translation of the Java class "Predict7"
# from the provided code. All functionality is intended to mirror the original
# Java version as closely as possible, line-by-line, preserving method names,
# logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for the UNDEFINED_FLOAT
#     constant. Make sure that file is imported in your Python environment.
#   - The usage example at the bottom (showing how to import this file into
#     scenario_model.py) is abridged only. Everything else is full and
#     unabridged.
# =============================================================================

from scenario_constants import UNDEFINED_FLOAT

class Predict7:
    """
    A container for scenario predicted miscellaneous variables, including:
    fluid loss, heart rate, stroke volume, O2 debt, and physiological strain index (PSI).
    Translated from 'Predict7.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all variables to UNDEFINED_FLOAT.
        Mirrors: public Predict7()
        """
        self.fluidLoss = UNDEFINED_FLOAT  # [gm]
        self.HR = UNDEFINED_FLOAT         # [bpm]
        self.SV = UNDEFINED_FLOAT         # [ml]
        self.O2debt = UNDEFINED_FLOAT
        self.PSI = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict7 a copy of that Predict7.
        Mirrors: public void copyOf(Predict7 that)
        """
        self.fluidLoss = that.fluidLoss
        self.HR = that.HR
        self.SV = that.SV
        self.O2debt = that.O2debt
        self.PSI = that.PSI

    def reset(self):
        """
        Sets all variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.fluidLoss = UNDEFINED_FLOAT
        self.HR = UNDEFINED_FLOAT
        self.SV = UNDEFINED_FLOAT
        self.O2debt = UNDEFINED_FLOAT
        self.PSI = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getFluidLoss(self):
        """Returns the fluid loss [gm]."""
        return self.fluidLoss

    def getHR(self):
        """Returns the heart rate [bpm]."""
        return self.HR

    def getSV(self):
        """Returns the stroke volume [ml]."""
        return self.SV

    def getO2debt(self):
        """Returns the O2 debt."""
        return self.O2debt

    def getPSI(self):
        """Returns the physiological strain index."""
        return self.PSI

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setFluidLoss(self, fluidLoss):
        """Sets the fluid loss [gm]."""
        self.fluidLoss = fluidLoss

    def setHR(self, HR):
        """Sets the heart rate [bpm]."""
        self.HR = HR

    def setSV(self, SV):
        """Sets the stroke volume [ml]."""
        self.SV = SV

    def setO2debt(self, O2debt):
        """Sets the oxygen debt."""
        self.O2debt = O2debt

    def setPSI(self, PSI):
        """Sets the physiological strain index."""
        self.PSI = PSI


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.predict7 import Predict7
#
#   p7 = Predict7()
#   p7.setHR(140.0)
#   print("Heart Rate:", p7.getHR())
#
# =============================================================================
