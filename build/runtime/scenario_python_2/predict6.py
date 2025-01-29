# =============================================================================
# predict6.py
#
# This Python module is a direct translation of the Java class "Predict6"
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

class Predict6:
    """
    A container for scenario predicted miscellaneous variables.
    Translated from 'Predict6.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all miscellaneous variables to UNDEFINED_FLOAT.
        Mirrors: public Predict6()
        """
        self.SR     = UNDEFINED_FLOAT
        self.Drip   = UNDEFINED_FLOAT
        self.Esk    = UNDEFINED_FLOAT
        self.Emax   = UNDEFINED_FLOAT
        self.SkWet  = UNDEFINED_FLOAT
        self.dMshiv = UNDEFINED_FLOAT
        self.MRC    = UNDEFINED_FLOAT
        self.CrSkdT = UNDEFINED_FLOAT
        self.RaSkdT = UNDEFINED_FLOAT

    def copyOf(self, that):
        """
        Makes this Predict6 a copy of that Predict6.
        Mirrors: public void copyOf(Predict6 that)
        """
        self.SR     = that.SR
        self.Drip   = that.Drip
        self.Esk    = that.Esk
        self.Emax   = that.Emax
        self.SkWet  = that.SkWet
        self.dMshiv = that.dMshiv
        self.MRC    = that.MRC
        self.CrSkdT = that.CrSkdT
        self.RaSkdT = that.RaSkdT

    def reset(self):
        """
        Sets all miscellaneous variables to UNDEFINED_FLOAT.
        Mirrors: public void reset()
        """
        self.SR     = UNDEFINED_FLOAT
        self.Drip   = UNDEFINED_FLOAT
        self.Esk    = UNDEFINED_FLOAT
        self.Emax   = UNDEFINED_FLOAT
        self.SkWet  = UNDEFINED_FLOAT
        self.dMshiv = UNDEFINED_FLOAT
        self.MRC    = UNDEFINED_FLOAT
        self.CrSkdT = UNDEFINED_FLOAT
        self.RaSkdT = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getSR(self):
        """Returns the sweat rate [gm/min]."""
        return self.SR

    def getDrip(self):
        """Returns the drip rate [gm/min]."""
        return self.Drip

    def getEsk(self):
        """Returns the Esk value [W/m^2]."""
        return self.Esk

    def getEmax(self):
        """Returns the Emax value [W/m^2]."""
        return self.Emax

    def getSkWet(self):
        """Returns the skin percent wet [%]."""
        return self.SkWet

    def getdMshiv(self):
        """Returns the dMshiv value."""
        return self.dMshiv

    def getMRC(self):
        """Returns the MRC value."""
        return self.MRC

    def getCrSkdT(self):
        """Returns the core-skin temperature delta [C]."""
        return self.CrSkdT

    def getRaSkdT(self):
        """Returns the blood-skin temperature delta [C]."""
        return self.RaSkdT

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setSR(self, SR):
        """Sets the sweat rate [gm/min]."""
        self.SR = SR

    def setDrip(self, Drip):
        """Sets the drip rate [gm/min]."""
        self.Drip = Drip

    def setEsk(self, Esk):
        """Sets the Esk [W/m^2]."""
        self.Esk = Esk

    def setEmax(self, Emax):
        """Sets the Emax [W/m^2]."""
        self.Emax = Emax

    def setSkWet(self, SkWet):
        """Sets the skin percent wet [%]."""
        self.SkWet = SkWet

    def setdMshiv(self, dMshiv):
        """Sets the dMshiv value."""
        self.dMshiv = dMshiv

    def setMRC(self, MRC):
        """Sets the MRC value."""
        self.MRC = MRC

    def setCrSkdT(self, CrSkdT):
        """Sets the core-skin temperature delta [C]."""
        self.CrSkdT = CrSkdT

    def setRaSkdT(self, RaSkdT):
        """Sets the blood-skin temperature delta [C]."""
        self.RaSkdT = RaSkdT


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.predict6 import Predict6
#
#   p6 = Predict6()
#   p6.setSR(1.2)
#   print("Sweat rate:", p6.getSR())
#
# =============================================================================
