# =============================================================================
# environ_record.py
#
# This Python module is a direct translation of the Java class "EnvironRecord"
# from the provided code. It holds environmental attributes such as air
# temperature, wind speed, humidity, etc. All functionality is intended to
# mirror the original Java version as closely as possible, line-by-line,
# preserving method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for constants like
#     UNDEFINED_FLOAT, UNDEFINED_INT, etc. 
#   - It also depends on a utility function `Utils.SatVP(...)` from `utils.py`,
#     used to compute saturation vapor pressure. 
#   - The usage example at the bottom is abridged only; everything else is
#     full and unabridged.
# =============================================================================

from scenario_constants import (
    UNDEFINED_FLOAT, COMMA_STRING
)
from utils import SatVP

class EnvironRecord:
    """
    Holds environmental attributes such as air temperature, wind speed,
    relative humidity, and so on. Mirrors 'EnvironRecord.java'.
    """

    # -------------------------------------------------------------------------
    # Public static final constants (from Java)
    # -------------------------------------------------------------------------
    STANDARD_T_AIR = 26.0     # deg C
    STANDARD_T_MR  = 26.0     # deg C
    STANDARD_T_G   = 26.0     # deg C
    STANDARD_P_VAP = 7.0      # Torr (30% RH @ 25 C)
    STANDARD_V_AIR = 0.1      # m/s (still air)

    # column indexes
    TIME   = 1
    T_AIR  = 2
    T_MR   = 3
    T_G    = 4
    RH     = 5
    V_AIR  = 6
    NUM_FIELDS = 6

    colIdentifiers = [
        " Time ",
        " Tair ",
        " Tmr ",
        " Tg ",
        " RH ",
        " Vair "
    ]

    longValues = [
        float(10000),  # TIME
        float(10000),  # T_AIR
        float(10000),  # T_MR
        float(10000),  # T_G
        float(10000),  # RH
        float(10000)   # V_AIR
    ]

    toolTips = [
        "observation time (min)",
        "air temperature (C)",
        "mean radiant temperature (C)",
        "black globe temperature (C)",
        "relative humidity (%)",
        "wind speed (m/s)"
    ]

    ALL_CELLS_EDITABLE = []

    def __init__(self, t=UNDEFINED_FLOAT,
                 Tair=UNDEFINED_FLOAT, Tmr=UNDEFINED_FLOAT,
                 Tg=UNDEFINED_FLOAT, rh=UNDEFINED_FLOAT, Vair=UNDEFINED_FLOAT):
        """
        Mirrors the Java constructors:
          - public EnvironRecord()
          - public EnvironRecord(float t, float Tair, float Tmr, float Tg, float rh, float Vair)
          - public EnvironRecord(float t, float Tair, float rh, float Vair)
        This single __init__ merges them by allowing optional arguments.
        """
        # instance variables
        self.t    = t
        self.Tair = Tair
        self.Tmr  = Tmr
        self.Tg   = Tg
        self.rh   = rh
        self.Vair = Vair

        # not used currently, but present in Java
        self.Tdp  = UNDEFINED_FLOAT
        self.Twb  = UNDEFINED_FLOAT
        self.Tnwb = UNDEFINED_FLOAT
        self.Pvap = UNDEFINED_FLOAT

    def copyOf(self, environRecord):
        """
        Makes a copy of the specified object.
        Mirrors: public void copyOf(EnvironRecord environRecord).
        """
        self.t    = environRecord.get_t()
        self.Tair = environRecord.getTair()
        self.Tmr  = environRecord.getTmr()
        self.Tg   = environRecord.getTg()
        self.rh   = environRecord.getRH()
        self.Vair = environRecord.getVair()
        self.Tdp  = environRecord.getTdp()
        self.Twb  = environRecord.getTwb()
        self.Tnwb = environRecord.getTnwb()
        self.Pvap = environRecord.getPvap()

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_t(self):
        """Returns the time in [min]."""
        return self.t

    def getTair(self):
        """Returns the air temperature [C]."""
        return self.Tair

    def getTmr(self):
        """Returns the mean radiant temperature [C]."""
        return self.Tmr

    def getTg(self):
        """Returns the black globe temperature [C]."""
        return self.Tg

    def getRH(self):
        """Returns the relative humidity [%]."""
        return self.rh

    def getVair(self):
        """Returns the wind speed [m/s]."""
        return self.Vair

    def getPvap(self):
        """Returns the vapor pressure."""
        return self.Pvap

    def getTdp(self):
        """Returns the Tdp value."""
        return self.Tdp

    def getTwb(self):
        """Returns the Twb value."""
        return self.Twb

    def getTnwb(self):
        """Returns the Tnwb value."""
        return self.Tnwb

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Sets values to default constructor values.
        Mirrors: public void reset()
        """
        self.t    = UNDEFINED_FLOAT
        self.Tair = UNDEFINED_FLOAT
        self.Tmr  = UNDEFINED_FLOAT
        self.Tg   = UNDEFINED_FLOAT
        self.rh   = UNDEFINED_FLOAT
        self.Vair = UNDEFINED_FLOAT

        self.Tdp  = UNDEFINED_FLOAT
        self.Twb  = UNDEFINED_FLOAT
        self.Tnwb = UNDEFINED_FLOAT

        self.Pvap = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def set_t(self, t):
        """Sets the observation time [min]."""
        self.t = t

    def setTair(self, Tair):
        """Sets the air temperature [C]."""
        self.Tair = Tair

    def setTmr(self, Tmr):
        """Sets the mean radiant temperature [C]."""
        self.Tmr = Tmr

    def setTg(self, Tg):
        """Sets the black globe temperature [C]."""
        self.Tg = Tg

    def setRH(self, rh):
        """Sets the relative humidity [%]."""
        self.rh = rh

    def setVair(self, Vair):
        """Sets the wind speed [m/s]."""
        self.Vair = Vair

    def setPvap(self, Pvap):
        """Sets the vapor pressure."""
        self.Pvap = Pvap

    def setTdp(self, Tdp):
        """Sets the Tdp value."""
        self.Tdp = Tdp

    def setTwb(self, Twb):
        """Sets the Twb value."""
        self.Twb = Twb

    def setTnwb(self, Tnwb):
        """Sets the Tnwb value."""
        self.Tnwb = Tnwb

    # -------------------------------------------------------------------------
    # toString
    # -------------------------------------------------------------------------
    def __str__(self):
        """
        Builds a comma separated string that consists of all data in this class.
        Mirrors: public String toString().
        """
        # We'll replicate logic from Java
        from scenario_python_2.scenario_constants import UNDEFINED_FLOAT

        if self.t != UNDEFINED_FLOAT:
            line = str(self.t)
        else:
            line = str(UNDEFINED_FLOAT)

        if self.Tair != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.Tair)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.Tmr != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.Tmr)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.Tg != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.Tg)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.rh != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.rh)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.Vair != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.Vair)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        return line

    # -------------------------------------------------------------------------
    # computeMissingValues
    # -------------------------------------------------------------------------
    def computeMissingValues(self):
        """
        Computes the essential data that may still be UNDEFINED.
        Mirrors: public void computeMissingValues().
        """
        # If Vair < .1f, set it to .1f
        if self.Vair == UNDEFINED_FLOAT or self.Vair < 0.1:
            self.Vair = 0.1

        # The code logic
        if (self.Tg == UNDEFINED_FLOAT and 
            self.Tmr != UNDEFINED_FLOAT and 
            self.Tair != UNDEFINED_FLOAT):
            k = 1.824 * (self.Vair ** 0.5)
            self.Tg = (self.Tmr + k * self.Tair) / (1 + k)

        if (self.Tmr == UNDEFINED_FLOAT and 
            self.Tg != UNDEFINED_FLOAT and
            self.Tair != UNDEFINED_FLOAT):
            k = 1.824 * (self.Vair ** 0.5)
            self.Tmr = self.Tg + k * (self.Tg - self.Tair)

        if (self.Tair != UNDEFINED_FLOAT and 
            self.Tmr == UNDEFINED_FLOAT):
            self.Tmr = self.Tair

        if (self.Tair != UNDEFINED_FLOAT and 
            self.Tg == UNDEFINED_FLOAT):
            self.Tg = self.Tair

        # still no Tair
        if self.Tair == UNDEFINED_FLOAT:
            self.Tair = self.STANDARD_T_AIR
            self.Tmr  = self.STANDARD_T_MR
            self.Tg   = self.STANDARD_T_G

        if self.Pvap == UNDEFINED_FLOAT:
            # we can compute if rh is known
            if self.rh != UNDEFINED_FLOAT:
                self.Pvap = self.rh * SatVP(self.Tair) / 100.0
            elif self.Tdp != UNDEFINED_FLOAT:
                self.Pvap = SatVP(self.Tdp)
            elif self.Twb != UNDEFINED_FLOAT:
                # Twb eqn from Java: SatVP(Twb) - .674825 * (Tair - Twb)
                self.Pvap = SatVP(self.Twb) - 0.674825 * (self.Tair - self.Twb)
            elif self.Tnwb != UNDEFINED_FLOAT:
                # Java code is incomplete for Tnwb logic, but let's replicate
                # the approach:
                # Twb = Tnwb - .5f + (Vair <= 1) - .13f * (Tg - Tair);
                # self.Pvap = SatVP(Twb) - ...
                # However, the code is commented "don't know what this line means"
                # We'll replicate what's there:
                # "Pvap = SatVP(Twb) - .674825*(Tair - Twb);"
                self.Pvap = self.STANDARD_P_VAP  # or replicate partial logic
                # We do as Java does, which sets self.Pvap to the same formula:
                # But the Java code references Twb, which is not redefined,
                # so we do the same approach:
                self.Pvap = SatVP(self.Twb) - 0.674825 * (self.Tair - self.Twb)
            else:
                self.Pvap = self.STANDARD_P_VAP

    # -------------------------------------------------------------------------
    # Class-level metadata methods (for table columns, etc.)
    # -------------------------------------------------------------------------
    @classmethod
    def getColumnCount(cls):
        """Gets the number of columns."""
        return len(cls.colIdentifiers)

    @classmethod
    def getColumnIdentifiers(cls):
        """Gets the column identifier strings."""
        return cls.colIdentifiers

    @classmethod
    def getLongValues(cls):
        """Gets the array of column data type placeholders."""
        return cls.longValues

    @classmethod
    def getToolTips(cls):
        """Gets the tool tips for each column header."""
        return cls.toolTips

    @classmethod
    def getUnEditableOrdinals(cls):
        """Gets the uneditable ordinals."""
        return cls.ALL_CELLS_EDITABLE

    # -------------------------------------------------------------------------
    # saveValue / getValue
    # -------------------------------------------------------------------------
    def saveValue(self, index, value):
        """
        Saves the specified value at the given index.
        Mirrors: public void saveValue(int index, Object value).
        """
        if index == self.TIME:
            self.set_t(float(value))
        elif index == self.T_AIR:
            self.setTair(float(value))
        elif index == self.T_MR:
            self.setTmr(float(value))
        elif index == self.T_G:
            self.setTg(float(value))
        elif index == self.RH:
            self.setRH(float(value))
        elif index == self.V_AIR:
            self.setVair(float(value))

    def getValue(self, index):
        """
        Gets the value at the given column index.
        Mirrors: public Object getValue(int index).
        """
        if index == self.TIME:
            return float(self.t)
        elif index == self.T_AIR:
            return float(self.Tair)
        elif index == self.T_MR:
            return float(self.Tmr)
        elif index == self.T_G:
            return float(self.Tg)
        elif index == self.RH:
            return float(self.rh)
        elif index == self.V_AIR:
            return float(self.Vair)
        return None


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class:
#
#   from scenario_python_2.environ_record import EnvironRecord
#
#   er = EnvironRecord(0.0, 25.0, 25.0, 25.0, 50.0, 0.5)
#   er.computeMissingValues()
#   print(er)
#
# =============================================================================
