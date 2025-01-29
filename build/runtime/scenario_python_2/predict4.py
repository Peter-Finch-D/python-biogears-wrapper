# =============================================================================
# predict4.py
#
# This Python module is a direct translation of the Java class "Predict4"
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

class Predict4:
    """
    A container for scenario predicted specific heat fluxes [W/m^2].
    Translated from 'Predict4.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all heat flux variables to UNDEFINED_FLOAT.
        Mirrors: public Predict4()
        """
        self.dQtot_dt = UNDEFINED_FLOAT
        self.dQra_dt  = UNDEFINED_FLOAT
        self.dQcr_dt  = UNDEFINED_FLOAT
        self.dQmu_dt  = UNDEFINED_FLOAT
        self.dQfat_dt = UNDEFINED_FLOAT
        self.dQvsk_dt = UNDEFINED_FLOAT
        self.dQsk_dt  = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict4 a copy of that Predict4.
        Mirrors: public void copyOf(Predict4 that)
        """
        self.dQtot_dt = that.dQtot_dt
        self.dQra_dt  = that.dQra_dt
        self.dQcr_dt  = that.dQcr_dt
        self.dQmu_dt  = that.dQmu_dt
        self.dQfat_dt = that.dQfat_dt
        self.dQvsk_dt = that.dQvsk_dt
        self.dQsk_dt  = that.dQsk_dt

    def reset(self):
        """
        Sets all heat flux variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.dQtot_dt = UNDEFINED_FLOAT
        self.dQra_dt  = UNDEFINED_FLOAT
        self.dQcr_dt  = UNDEFINED_FLOAT
        self.dQmu_dt  = UNDEFINED_FLOAT
        self.dQfat_dt = UNDEFINED_FLOAT
        self.dQvsk_dt = UNDEFINED_FLOAT
        self.dQsk_dt  = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getdQtot_dt(self):
        """Returns the total heat flux [W/m^2]."""
        return self.dQtot_dt

    def getdQra_dt(self):
        """Returns the blood heat flux [W/m^2]."""
        return self.dQra_dt

    def getdQcr_dt(self):
        """Returns the core heat flux [W/m^2]."""
        return self.dQcr_dt

    def getdQmu_dt(self):
        """Returns the muscle heat flux [W/m^2]."""
        return self.dQmu_dt

    def getdQfat_dt(self):
        """Returns the fat heat flux [W/m^2]."""
        return self.dQfat_dt

    def getdQvsk_dt(self):
        """Returns the vascular skin heat flux [W/m^2]."""
        return self.dQvsk_dt

    def getdQsk_dt(self):
        """Returns the skin heat flux [W/m^2]."""
        return self.dQsk_dt

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setdQtot_dt(self, dQtot_dt):
        """Sets the total heat flux [W/m^2]."""
        self.dQtot_dt = dQtot_dt

    def setdQra_dt(self, dQra_dt):
        """Sets the blood heat flux [W/m^2]."""
        self.dQra_dt = dQra_dt

    def setdQcr_dt(self, dQcr_dt):
        """Sets the core heat flux [W/m^2]."""
        self.dQcr_dt = dQcr_dt

    def setdQmu_dt(self, dQmu_dt):
        """Sets the muscle heat flux [W/m^2]."""
        self.dQmu_dt = dQmu_dt

    def setdQfat_dt(self, dQfat_dt):
        """Sets the fat heat flux [W/m^2]."""
        self.dQfat_dt = dQfat_dt

    def setdQvsk_dt(self, dQvsk_dt):
        """Sets the vascular skin heat flux [W/m^2]."""
        self.dQvsk_dt = dQvsk_dt

    def setdQsk_dt(self, dQsk_dt):
        """Sets the skin heat flux [W/m^2]."""
        self.dQsk_dt = dQsk_dt


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.predict4 import Predict4
#
#   p4 = Predict4()
#   p4.setdQtot_dt(95.0)
#   print("Total heat flux:", p4.getdQtot_dt())
#
# =============================================================================
