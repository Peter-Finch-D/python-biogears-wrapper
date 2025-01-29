# =============================================================================
# predict1.py
#
# This Python module is a direct translation of the Java class "Predict1"
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

class Predict1:
    """
    A container for scenario predicted temperatures.
    Translated from 'Predict1.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all temperature variables to UNDEFINED_FLOAT.
        Mirrors: public Predict1()
        """
        self.Tra  = UNDEFINED_FLOAT
        self.Tcr  = UNDEFINED_FLOAT
        self.Tmu  = UNDEFINED_FLOAT
        self.Tfat = UNDEFINED_FLOAT
        self.Tvsk = UNDEFINED_FLOAT
        self.Tsk  = UNDEFINED_FLOAT
        self.Tcl  = UNDEFINED_FLOAT
        self.Tbdy = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict1 a copy of that Predict1.
        Mirrors: public void copyOf(Predict1 that)
        """
        self.Tra  = that.Tra
        self.Tcr  = that.Tcr
        self.Tmu  = that.Tmu
        self.Tfat = that.Tfat
        self.Tvsk = that.Tvsk
        self.Tsk  = that.Tsk
        self.Tcl  = that.Tcl
        self.Tbdy = that.Tbdy

    def reset(self):
        """
        Sets all temperature variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.Tra  = UNDEFINED_FLOAT
        self.Tcr  = UNDEFINED_FLOAT
        self.Tmu  = UNDEFINED_FLOAT
        self.Tfat = UNDEFINED_FLOAT
        self.Tvsk = UNDEFINED_FLOAT
        self.Tsk  = UNDEFINED_FLOAT
        self.Tcl  = UNDEFINED_FLOAT
        self.Tbdy = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getTra(self):
        """Returns the blood temperature [C]."""
        return self.Tra

    def getTcr(self):
        """Returns the core temperature [C]."""
        return self.Tcr

    def getTmu(self):
        """Returns the muscle temperature [C]."""
        return self.Tmu

    def getTfat(self):
        """Returns the fat temperature [C]."""
        return self.Tfat

    def getTvsk(self):
        """Returns the vascular skin temperature [C]."""
        return self.Tvsk

    def getTsk(self):
        """Returns the skin temperature [C]."""
        return self.Tsk

    def getTcl(self):
        """Returns the clothing temperature [C]."""
        return self.Tcl

    def getTbdy(self):
        """Returns the body temperature [C]."""
        return self.Tbdy

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setTra(self, Tra):
        """Sets the blood temperature [C]."""
        self.Tra = Tra

    def setTcr(self, Tcr):
        """Sets the core temperature [C]."""
        self.Tcr = Tcr

    def setTmu(self, Tmu):
        """Sets the muscle temperature [C]."""
        self.Tmu = Tmu

    def setTfat(self, Tfat):
        """Sets the fat temperature [C]."""
        self.Tfat = Tfat

    def setTvsk(self, Tvsk):
        """Sets the vascular skin temperature [C]."""
        self.Tvsk = Tvsk

    def setTsk(self, Tsk):
        """Sets the skin temperature [C]."""
        self.Tsk = Tsk

    def setTcl(self, Tcl):
        """Sets the clothing temperature [C]."""
        self.Tcl = Tcl

    def setTbdy(self, Tbdy):
        """Sets the body temperature [C]."""
        self.Tbdy = Tbdy


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.predict1 import Predict1
#
#   p1 = Predict1()
#   p1.setTra(37.0)
#   print("Blood temp:", p1.getTra())
#
# =============================================================================
