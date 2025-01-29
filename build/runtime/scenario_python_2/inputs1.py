# =============================================================================
# inputs1.py
#
# This Python module is a direct translation of the Java class "Inputs1"
# from the provided code. It stores scenario subject, clothing, and environmental
# inputs. All functionality is intended to mirror the original Java version
# as closely as possible, line-by-line, preserving method names, logic, and
# comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for the constants
#     UNDEFINED_FLOAT and UNDEFINED_INT.
#   - It also references "EnvironRecord", "ClothingRecord", and "SubjectRecord"
#     standard values (e.g., STANDARD_T_AIR, STANDARD_WEIGHT, etc.). In Python,
#     you must define or mock them in corresponding modules or handle them
#     as needed in your environment.
#   - The usage example at the bottom (showing how to import this file) is
#     abridged only. Everything else is full and unabridged.
# =============================================================================

from scenario_constants import UNDEFINED_FLOAT, UNDEFINED_INT

# If you have modules named environ_record, clothing_record, subject_record
# that define the "STANDARD_..." constants, import them here:
from environ_record import EnvironRecord
from clothing_record import ClothingRecord
from subject_record import SubjectRecord
#
# For example, they might provide:
# EnvironRecord.STANDARD_T_AIR = 25.0
# ClothingRecord.STANDARD_ICLO = 0.6
# SubjectRecord.STANDARD_WEIGHT = 70.0
#
# If you do not have such classes, you can stub them or define them as needed.

class Inputs1:
    """
    A container for scenario subject, clothing, and environmental inputs.
    Translated from 'Inputs1.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all values to UNDEFINED_FLOAT or UNDEFINED_INT.
        Mirrors: public Inputs1().
        """
        # environment
        self.Ta   = UNDEFINED_FLOAT  # air temperature [C]
        self.Tmr  = UNDEFINED_FLOAT  # mean radiant temperature [C]
        self.Tg   = UNDEFINED_FLOAT  # globe temperature [C]
        self.Pvap = UNDEFINED_FLOAT  # vapor pressure [Torr]
        self.Vair = UNDEFINED_FLOAT  # wind speed [m/s]

        # clothing
        self.Iclo = UNDEFINED_FLOAT  # clothing insulation value (dimensionless)
        self.Im   = UNDEFINED_FLOAT  # clothing permeability index (dimensionless)

        # subject
        self.BW   = UNDEFINED_FLOAT  # body weight [kg]
        self.HT   = UNDEFINED_FLOAT  # body height [cm]
        self.SA   = UNDEFINED_FLOAT  # DuBois surface area [m^2]
        self.AGE  = UNDEFINED_INT    # subject age [years]
        self.PctFat = UNDEFINED_FLOAT # subject body fat [%]

    def copyOf(self, that):
        """
        Makes this Inputs1 a copy of that Inputs1.
        Mirrors: public void copyOf(Inputs1 that)
        """
        self.Ta   = that.Ta
        self.Tmr  = that.Tmr
        self.Tg   = that.Tg
        self.Pvap = that.Pvap
        self.Vair = that.Vair

        self.Iclo = that.Iclo
        self.Im   = that.Im

        self.BW   = that.BW
        self.HT   = that.HT
        self.SA   = that.SA
        self.AGE  = that.AGE
        self.PctFat = that.PctFat

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getTa(self):
        """Returns the air temperature [C]."""
        return self.Ta

    def getTmr(self):
        """Returns the mean radiant temperature [C]."""
        return self.Tmr

    def getTg(self):
        """Returns the globe temperature [C]."""
        return self.Tg

    def getPvap(self):
        """Returns the vapor pressure [Torr]."""
        return self.Pvap

    def getVair(self):
        """Returns the wind speed [m/s]."""
        return self.Vair

    def getIclo(self):
        """Returns the Iclo clothing value (dimensionless)."""
        return self.Iclo

    def getIm(self):
        """Returns the Im clothing value (dimensionless)."""
        return self.Im

    def getBW(self):
        """Returns the subject's weight [kg]."""
        return self.BW

    def getHT(self):
        """Returns the subject's height [cm]."""
        return self.HT

    def getSA(self):
        """Returns the subject's surface area [m^2]."""
        return self.SA

    def getAGE(self):
        """Returns the subject's age [years]."""
        return self.AGE

    def getPctFat(self):
        """Returns the subject's body fat [%]."""
        return self.PctFat

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setTa(self, Ta):
        """Sets the air temperature [C]."""
        self.Ta = Ta

    def setTmr(self, Tmr):
        """Sets the mean radiant temperature [C]."""
        self.Tmr = Tmr

    def setTg(self, Tg):
        """Sets the globe temperature [C]."""
        self.Tg = Tg

    def setPvap(self, Pvap):
        """Sets the vapor pressure [Torr]."""
        self.Pvap = Pvap

    def setVair(self, Vair):
        """Sets the wind speed [m/s]."""
        self.Vair = Vair

    def setIclo(self, Iclo):
        """Sets the Iclo clothing value (dimensionless)."""
        self.Iclo = Iclo

    def setIm(self, Im):
        """Sets the Im clothing value (dimensionless)."""
        self.Im = Im

    def setBW(self, BW):
        """Sets the subject's weight [kg]."""
        self.BW = BW

    def setHT(self, HT):
        """Sets the subject's height [cm]."""
        self.HT = HT

    def setSA(self, SA):
        """Sets the subject's surface area [m^2]."""
        self.SA = SA

    def setAGE(self, AGE):
        """Sets the subject's age [years]."""
        self.AGE = AGE

    def setPctFat(self, PctFat):
        """Sets the subject's body fat [%]."""
        self.PctFat = PctFat

    # -------------------------------------------------------------------------
    # Resetting methods
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Helper function to set all Inputs1 data to 'undefined' values.
        Mirrors: public void reset()
        """
        self.resetEnviron()
        self.resetClothing()
        self.resetSubject()

    def resetEnviron(self):
        """
        Sets all environment variables to 'undefined' values.
        Mirrors: public void resetEnviron()
        """
        self.Ta   = UNDEFINED_FLOAT
        self.Tmr  = UNDEFINED_FLOAT
        self.Tg   = self.Tmr
        self.Pvap = UNDEFINED_FLOAT
        self.Vair = UNDEFINED_FLOAT

    def resetClothing(self):
        """
        Sets all clothing variables to 'undefined' values.
        Mirrors: public void resetClothing()
        """
        self.Iclo = UNDEFINED_FLOAT
        self.Im   = UNDEFINED_FLOAT

    def resetSubject(self):
        """
        Sets all subject data to 'undefined' values.
        Mirrors: public void resetSubject()
        """
        self.BW     = UNDEFINED_FLOAT
        self.HT     = UNDEFINED_FLOAT
        self.SA     = UNDEFINED_FLOAT
        self.AGE    = UNDEFINED_INT
        self.PctFat = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Standard values methods
    # -------------------------------------------------------------------------
    def setStandardValues(self):
        """
        Helper function to set all Inputs1 data to standard values.
        Mirrors: public void setStandardValues()
        """
        self.setStandardEnviron()
        self.setStandardClothing()
        self.setStandardSubject()

    def setStandardEnviron(self):
        """
        Sets all environment variables to standard values.
        Mirrors: public void setStandardEnviron()
        """
        # This code references EnvironRecord.STANDARD_T_AIR, etc.
        # In Python, you need to define those if they aren't already.
        # For example:
        #   from scenario_python_2.environ_record import EnvironRecord
        #   self.Ta   = EnvironRecord.STANDARD_T_AIR
        # We'll just place placeholders here:
        self.Ta   = EnvironRecord.STANDARD_T_AIR
        self.Tmr  = EnvironRecord.STANDARD_T_MR
        self.Tg   = EnvironRecord.STANDARD_T_G
        self.Pvap = EnvironRecord.STANDARD_P_VAP
        self.Vair = EnvironRecord.STANDARD_V_AIR

    def setStandardClothing(self):
        """
        Sets all clothing variables to standard values.
        Mirrors: public void setStandardClothing()
        """
        # This code references ClothingRecord.STANDARD_ICLO, etc.
        # We'll do the same pattern as above:
        self.Iclo = ClothingRecord.STANDARD_ICLO
        self.Im   = ClothingRecord.STANDARD_IM

    def setStandardSubject(self):
        """
        Sets all subject variables to standard values.
        Mirrors: public void setStandardSubject()
        """
        # This code references SubjectRecord.STANDARD_WEIGHT, etc.
        # We'll do the same pattern:
        self.BW     = SubjectRecord.STANDARD_WEIGHT
        self.HT     = SubjectRecord.STANDARD_HEIGHT
        self.SA     = SubjectRecord.STANDARD_SA
        self.AGE    = SubjectRecord.STANDARD_AGE
        self.PctFat = SubjectRecord.STANDARD_PCT_FAT

    def __str__(self):
        """
        Reports variable values (added in Java code on 06/03/2014).
        Mirrors: public String toString().
        """
        out = (f"Ta:{self.Ta} "
               f"Tmr:{self.Tmr} "
               f"Tg:{self.Tg} "
               f"Pvap:{self.Pvap} "
               f"Vair:{self.Vair} "
               f"Iclo:{self.Iclo} "
               f"Im:{self.Im} "
               f"BW:{self.BW} "
               f"HT:{self.HT} "
               f"SA:{self.SA} "
               f"Age:{self.AGE} "
               f"PctFat:{self.PctFat}")
        return out
