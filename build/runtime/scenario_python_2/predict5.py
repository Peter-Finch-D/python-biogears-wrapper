# =============================================================================
# predict5.py
#
# This Python module is a direct translation of the Java class "Predict5"
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

class Predict5:
    """
    A container for scenario predicted heat [kJ].
    Translated from 'Predict5.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all heat variables to UNDEFINED_FLOAT.
        Mirrors: public Predict5()
        """
        self.Qtot = UNDEFINED_FLOAT
        self.Qra  = UNDEFINED_FLOAT
        self.Qcr  = UNDEFINED_FLOAT
        self.Qmu  = UNDEFINED_FLOAT
        self.Qfat = UNDEFINED_FLOAT
        self.Qvsk = UNDEFINED_FLOAT
        self.Qsk  = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict5 a copy of that Predict5.
        Mirrors: public void copyOf(Predict5 that)
        """
        self.Qtot = that.Qtot
        self.Qra  = that.Qra
        self.Qcr  = that.Qcr
        self.Qmu  = that.Qmu
        self.Qfat = that.Qfat
        self.Qvsk = that.Qvsk
        self.Qsk  = that.Qsk

    def reset(self):
        """
        Sets all heat variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.Qtot = UNDEFINED_FLOAT
        self.Qra  = UNDEFINED_FLOAT
        self.Qcr  = UNDEFINED_FLOAT
        self.Qmu  = UNDEFINED_FLOAT
        self.Qfat = UNDEFINED_FLOAT
        self.Qvsk = UNDEFINED_FLOAT
        self.Qsk  = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getQtot(self):
        """Returns the total heat [kJ]."""
        return self.Qtot

    def getQra(self):
        """Returns the blood heat [kJ]."""
        return self.Qra

    def getQcr(self):
        """Returns the core heat [kJ]."""
        return self.Qcr

    def getQmu(self):
        """Returns the muscle heat [kJ]."""
        return self.Qmu

    def getQfat(self):
        """Returns the fat heat [kJ]."""
        return self.Qfat

    def getQvsk(self):
        """Returns the vascular skin heat [kJ]."""
        return self.Qvsk

    def getQsk(self):
        """Returns the skin heat [kJ]."""
        return self.Qsk

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setQtot(self, Qtot):
        """Sets the total heat [kJ]."""
        self.Qtot = Qtot

    def setQra(self, Qra):
        """Sets the blood heat [kJ]."""
        self.Qra = Qra

    def setQcr(self, Qcr):
        """Sets the core heat [kJ]."""
        self.Qcr = Qcr

    def setQmu(self, Qmu):
        """Sets the muscle heat [kJ]."""
        self.Qmu = Qmu

    def setQfat(self, Qfat):
        """Sets the fat heat [kJ]."""
        self.Qfat = Qfat

    def setQvsk(self, Qvsk):
        """Sets the vascular skin heat [kJ]."""
        self.Qvsk = Qvsk

    def setQsk(self, Qsk):
        """Sets the skin heat [kJ]."""
        self.Qsk = Qsk