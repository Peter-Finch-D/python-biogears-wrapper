# =============================================================================
# predict2.py
#
# This Python module is a direct translation of the Java class "Predict2"
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

class Predict2:
    """
    A container for scenario predicted blood flow rates.
    Translated from 'Predict2.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all blood flow variables to UNDEFINED_FLOAT.
        Mirrors: public Predict2()
        """
        self.BFra  = UNDEFINED_FLOAT
        self.BFcr  = UNDEFINED_FLOAT
        self.BFmu  = UNDEFINED_FLOAT
        self.BFfat = UNDEFINED_FLOAT
        self.BFvsk = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict2 a copy of that Predict2.
        Mirrors: public void copyOf(Predict2 that)
        """
        self.BFra  = that.BFra
        self.BFcr  = that.BFcr
        self.BFmu  = that.BFmu
        self.BFfat = that.BFfat
        self.BFvsk = that.BFvsk

    def reset(self):
        """
        Sets all blood flow variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.BFra  = UNDEFINED_FLOAT
        self.BFcr  = UNDEFINED_FLOAT
        self.BFmu  = UNDEFINED_FLOAT
        self.BFfat = UNDEFINED_FLOAT
        self.BFvsk = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getBFra(self):
        """Returns the total blood flow [cc/min]."""
        return self.BFra

    def getBFcr(self):
        """Returns the core blood flow [cc/min]."""
        return self.BFcr

    def getBFmu(self):
        """Returns the muscle blood flow [cc/min]."""
        return self.BFmu

    def getBFfat(self):
        """Returns the fat blood flow [cc/min]."""
        return self.BFfat

    def getBFvsk(self):
        """Returns the vascular skin blood flow [cc/min]."""
        return self.BFvsk

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setBFra(self, BFra):
        """Sets the total blood flow [cc/min]."""
        self.BFra = BFra

    def setBFcr(self, BFcr):
        """Sets the core blood flow [cc/min]."""
        self.BFcr = BFcr

    def setBFmu(self, BFmu):
        """Sets the muscle blood flow [cc/min]."""
        self.BFmu = BFmu

    def setBFfat(self, BFfat):
        """Sets the fat blood flow [cc/min]."""
        self.BFfat = BFfat

    def setBFvsk(self, BFvsk):
        """Sets the vascular skin blood flow [cc/min]."""
        self.BFvsk = BFvsk


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.predict2 import Predict2
#
#   p2 = Predict2()
#   p2.setBFra(5000.0)
#   print("Total blood flow:", p2.getBFra())
#
# =============================================================================
