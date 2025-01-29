# =============================================================================
# set_pts_n_flags.py
#
# This Python module is a direct translation of the Java class "SetPts_n_Flags"
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

DEFAULT_STR_TIME_OF_DAY = "00:00"

class SetPts_n_Flags:
    """
    A container for set points and alarm flags, mirroring the Java class
    'SetPts_n_Flags'. This includes:
      - hypothalamic set point temperature (ThypSet)
      - skin set point temperature (TskSet)
      - core temperature alarm (TcrFlag)
      - heart rate alarm (HRFlag)
      - heat flux alarm (dQ_dtFlag)
      - total heat alarm (QtotFlag)
      - time of day (timeOfDay)
      - string representation of time of day (strTimeOfDay)
    """

    def __init__(self,
                 ThypSet=UNDEFINED_FLOAT,
                 TskSet=UNDEFINED_FLOAT,
                 TcrFlag=UNDEFINED_FLOAT,
                 HRFlag=UNDEFINED_FLOAT,
                 dQ_dtFlag=UNDEFINED_FLOAT,
                 QtotFlag=UNDEFINED_FLOAT,
                 timeOfDay=UNDEFINED_FLOAT):
        """
        Mirrors the Java constructor behavior:
          - If no arguments, sets all to UNDEFINED_FLOAT
          - If arguments provided, sets them accordingly

        The second constructor in Java was:
            public SetPts_n_Flags(float ThypSet, float TskSet, ...)
        This Python __init__ merges both constructors by allowing optional params.
        """
        self.ThypSet = ThypSet
        self.TskSet = TskSet
        self.TcrFlag = TcrFlag
        self.HRFlag = HRFlag
        self.dQ_dtFlag = dQ_dtFlag
        self.QtotFlag = QtotFlag
        self.timeOfDay = timeOfDay

        # The Java code constructs NumberFormat with min integer digits=2
        # In Python, we'll replicate that behavior in our 'formatTimeOfDay' method
        # by formatting hours, minutes as 2-digit strings.

        # Build initial time string
        if self.timeOfDay == UNDEFINED_FLOAT:
            self.strTimeOfDay = DEFAULT_STR_TIME_OF_DAY
        else:
            self.strTimeOfDay = self.formatTimeOfDay()

    def copyOf(self, that):
        """
        Makes this SetPts_n_Flags a copy of that SetPts_n_Flags.
        Mirrors: public void copyOf(SetPts_n_Flags that)
        """
        self.ThypSet = that.ThypSet
        self.TskSet = that.TskSet
        self.TcrFlag = that.TcrFlag
        self.HRFlag = that.HRFlag
        self.dQ_dtFlag = that.dQ_dtFlag
        self.QtotFlag = that.QtotFlag
        self.timeOfDay = that.timeOfDay
        # Create a new string with the same value
        self.strTimeOfDay = str(that.strTimeOfDay)

    def reset(self):
        """
        Sets all variables to "undefined" or default values for time of day.
        Mirrors: public void reset()
        """
        self.ThypSet = UNDEFINED_FLOAT
        self.TskSet = UNDEFINED_FLOAT
        self.TcrFlag = UNDEFINED_FLOAT
        self.HRFlag = UNDEFINED_FLOAT
        self.dQ_dtFlag = UNDEFINED_FLOAT
        self.QtotFlag = UNDEFINED_FLOAT
        self.timeOfDay = UNDEFINED_FLOAT
        self.strTimeOfDay = DEFAULT_STR_TIME_OF_DAY

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getThypSet(self):
        """Returns the hypothalamic set point temperature [C]."""
        return self.ThypSet

    def getTskSet(self):
        """Returns the skin set point temperature [C]."""
        return self.TskSet

    def getTcrFlag(self):
        """Returns the core temperature alarm [C]."""
        return self.TcrFlag

    def getHRFlag(self):
        """Returns the heart rate alarm [bpm]."""
        return self.HRFlag

    def getdQ_dtFlag(self):
        """Returns the heat flux alarm [W/m^2]."""
        return self.dQ_dtFlag

    def getQtotFlag(self):
        """Returns the total heat alarm [kJ]."""
        return self.QtotFlag

    def getTimeOfDay(self):
        """
        Returns the time of day (float 0-23.99).
        Mirrors: public float getTimeOfDay()
        """
        return self.timeOfDay

    def getStrTimeOfDay(self):
        """
        Returns the time of day as a string in hh:mm format.
        Mirrors: public String getStrTimeOfDay()
        """
        return self.strTimeOfDay

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setThypSet(self, ThypSet):
        """Sets the hypothalamic set point temperature [C]."""
        self.ThypSet = ThypSet

    def setTskSet(self, TskSet):
        """Sets the skin set point temperature [C]."""
        self.TskSet = TskSet

    def setTcrFlag(self, TcrFlag):
        """Sets the core temperature alarm [C]."""
        self.TcrFlag = TcrFlag

    def setHRFlag(self, HRFlag):
        """Sets the heart rate alarm [bpm]."""
        self.HRFlag = HRFlag

    def setdQ_dtFlag(self, dQ_dtFlag):
        """Sets the heat flux alarm [W/m^2]."""
        self.dQ_dtFlag = dQ_dtFlag

    def setQtotFlag(self, QtotFlag):
        """Sets the total heat alarm [kJ]."""
        self.QtotFlag = QtotFlag

    def setTimeOfDay(self, timeOfDay):
        """
        Sets the time of day (float). Also updates strTimeOfDay.
        Mirrors: public void setTimeOfDay(float timeOfDay)
        """
        self.timeOfDay = timeOfDay
        self.strTimeOfDay = self.formatTimeOfDay()

    # -------------------------------------------------------------------------
    # Private/Helper function
    # -------------------------------------------------------------------------
    def formatTimeOfDay(self):
        """
        Converts this instance's timeOfDay to a formatted string in hh:mm format,
        preserving the Java logic that ensures two-digit hours and minutes.
        Mirrors: String formatTimeOfDay() in Java
        """
        # If timeOfDay is UNDEFINED_FLOAT, just return the default
        if self.timeOfDay == UNDEFINED_FLOAT:
            return DEFAULT_STR_TIME_OF_DAY

        nHr = int(self.timeOfDay)
        frac = self.timeOfDay - float(nHr)
        nMn = int(frac * 60.0 + 0.5)
        if nMn == 60:
            nMn = 0
            nHr += 1
            if nHr == 24:
                nHr = 0

        # We want 2 digits for each, so we can use zfill(2)
        s1 = str(nHr).zfill(2)
        s2 = str(nMn).zfill(2)
        return f"{s1}:{s2}"
