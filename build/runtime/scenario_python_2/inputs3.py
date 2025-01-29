# =============================================================================
# inputs3.py
#
# This Python module is a direct translation of the Java class "Inputs3"
# from the provided code. It stores recent scenario input including:
# acclimation index, dehydration index, start time (for circadian temperature
# estimate), an optional core temperature override, and flags for circadian
# model and core temperature override. All functionality is intended to mirror
# the original Java version as closely as possible, line-by-line, preserving
# method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for UNDEFINED_FLOAT
#     and UNDEFINED_INT, and presumably more.
# =============================================================================

from scenario_constants import UNDEFINED_FLOAT, UNDEFINED_INT

class Inputs3:
    """
    A container for recent scenario input that includes:
      - acclimation index
      - dehydration index
      - start time (for circadian temperature estimate)
      - Tcr0 (initial core temperature override)
      - circadian model flag
      - tcoreOverride flag
    Translated from 'Inputs3.java'.
    """

    def __init__(self,
                 acclimIndex=UNDEFINED_INT,
                 dehydIndex=UNDEFINED_INT,
                 startTime=UNDEFINED_INT,
                 Tcr0=UNDEFINED_FLOAT,
                 circadianModel=False,
                 tcoreOverride=False):
        """
        Default constructor if no arguments, or full constructor if all arguments
        are provided. Mirrors the behavior in Java:
          public Inputs3() {}
          public Inputs3(int acclimIndex, int dehydIndex, ...)
        """
        self.acclimIndex = acclimIndex
        self.dehydIndex = dehydIndex
        self.startTime = startTime
        self.Tcr0 = Tcr0
        self.circadianModel = circadianModel
        self.tcoreOverride = tcoreOverride

    def copyOf(self, that):
        """
        Makes this Inputs3 a copy of that Inputs3.
        Mirrors: public void copyOf(Inputs3 that)
        """
        self.acclimIndex = that.acclimIndex
        self.dehydIndex = that.dehydIndex
        self.startTime = that.startTime
        self.Tcr0 = that.Tcr0
        self.circadianModel = that.circadianModel
        self.tcoreOverride = that.tcoreOverride

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getAcclimIndex(self):
        """Returns the acclimation index."""
        return self.acclimIndex

    def getDehydIndex(self):
        """Returns the dehydration index."""
        return self.dehydIndex

    def getStartTime(self):
        """Returns the start time (an integer 0-23)."""
        return self.startTime

    def getTcr0(self):
        """Returns the initial core temperature override [C]."""
        return self.Tcr0

    def getCircadianModel(self):
        """Returns the circadian model flag."""
        return self.circadianModel

    def getTcoreOverride(self):
        """Returns the initial core temperature override flag."""
        return self.tcoreOverride

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setAcclimIndex(self, acclimIndex):
        """Sets the acclimation index."""
        self.acclimIndex = acclimIndex

    def setDehydIndex(self, dehydIndex):
        """Sets the dehydration index."""
        self.dehydIndex = dehydIndex

    def setStartTime(self, startTime):
        """Sets the start time (an integer 0-23)."""
        self.startTime = startTime

    def setTcr0(self, Tcr0):
        """Sets the initial core temperature override [C]."""
        self.Tcr0 = Tcr0

    def setCircadianModel(self, circadianModel):
        """Sets the circadian model flag."""
        self.circadianModel = circadianModel

    def setTcoreOverride(self, tcoreOverride):
        """Sets the core temperature override flag."""
        self.tcoreOverride = tcoreOverride

    # -------------------------------------------------------------------------
    # Reset / Compute
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Sets all variables to 'undefined' or false values.
        Mirrors: public void reset()
        """
        self.acclimIndex = UNDEFINED_INT
        self.dehydIndex = UNDEFINED_INT
        self.startTime = UNDEFINED_INT
        self.Tcr0 = UNDEFINED_FLOAT
        self.circadianModel = False
        self.tcoreOverride = False

    def computeMissingValues(self):
        """
        Computes missing/undefined values. Currently does nothing in Java code.
        Mirrors: public void computeMissingValues()
        """
        pass

    def __str__(self):
        """
        Reports variable values. Mirrors the Java toString().
        """
        out = (f"acclimIndex:{self.acclimIndex} "
               f"dehydrateIndex:{self.dehydIndex} "
               f"startTime:{self.startTime} "
               f"Tcr0:{self.Tcr0} "
               f"circadianModel:{self.circadianModel} "
               f"tcoreOverride:{self.tcoreOverride}")
        return out


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class:
#
#   from scenario_python_2.inputs3 import Inputs3
#
#   i3 = Inputs3()
#   i3.setAcclimIndex(2)
#   i3.setDehydIndex(1)
#   i3.setStartTime(8)
#   i3.setTcr0(37.0)
#   i3.setCircadianModel(True)
#   i3.setTcoreOverride(False)
#   print(i3)
#
# =============================================================================
