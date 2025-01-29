# =============================================================================
# activity_record.py
#
# This Python module is a direct translation of the Java class "ActivityRecord"
# from the provided code. It holds attributes for the activity such as the walk
# speed, the weight of the load, metabolic rates, and so on. All functionality
# is intended to mirror the original Java version as closely as possible,
# line-by-line, preserving method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for constants such as
#     UNDEFINED_FLOAT, UNDEFINED_STRING, and SCENARIO_M_PER_SEC_TO_MILES_PER_HR,
#     SCENARIO_KG_TO_LBS, GM_PER_MIN_TO_L_PER_HR, COMMA_STRING, etc.
#   - If you have a different approach to how constants or other references
#     are defined, adjust accordingly.
#   - The usage example at the bottom is abridged only; everything else
#     is full and unabridged.
# =============================================================================

from scenario_constants import (
    UNDEFINED_FLOAT, UNDEFINED_STRING,
    SCENARIO_M_PER_SEC_TO_MILES_PER_HR,
    SCENARIO_KG_TO_LBS,
    GM_PER_MIN_TO_L_PER_HR,
    COMMA_STRING,
    EMPTY_STRING
)

class ActivityRecord:
    """
    Holds attributes for the activity, such as walk speed, load weight,
    metabolic rates, fluid intake, etc.
    Translated from 'ActivityRecord.java'.
    """

    # -------------------------------------------------------------------------
    # Public static final constants from Java
    # -------------------------------------------------------------------------
    STANDARD_WORKMODE = "r"
    STANDARD_WALKSPEED = 0.0
    STANDARD_LOAD = 0.0
    STANDARD_TERRAIN = 1.0
    STANDARD_GRADE = 0.0
    STANDARD_FLUID_INTAKE = 0.0

    # ordinal fields
    TIME         = 1
    WORK_MODE    = 2
    MTOT         = 3
    MEXT         = 4
    MREST        = 5
    WALK_SPEED   = 6
    LOAD         = 7
    TERRAIN      = 8
    GRADE        = 9
    CLOTHING     = 10
    FLUID_INTAKE = 11

    NUM_FIELDS = 11
    MIN_NUM_FIELDS = 6  # backward compatibility

    # column identifier strings
    colIdentifiers = [
        " Time ",
        " Work Mode ",
        " M. Total ",
        " M. Ext ",
        " M. Rest ",
        " Walk Speed ",
        " Load ",
        " Terrain ",
        " Grade ",
        " Clothing ",
        "Fluid Intake"
    ]

    # data type placeholders (Java used them for table column sizing)
    longValues = [
        float(10000),
        str(" xx "),
        float(10000),
        float(10000),
        float(10000),
        float(10000),
        float(10000),
        float(10000),
        float(10000),
        str(" WBDU "),
        float(10000)
    ]

    # tooltips for each column
    toolTips = [
        "observation time (min)",
        "work mode (\"r\", \"f\" \"m\")",
        "total metabolic rate (W)",
        "external work (W)",
        "resting metabolic rate (W)",
        "walking speed (m/s)",
        "load (kg)",
        "terrain factor",
        "grade (%)",
        "clothing",
        "fluid intake (gm/min)"
    ]

    # If the Java code references uneditable columns, it returns "ALL_CELLS_EDITABLE"
    # or something similar. We'll define an array, but by default we assume all are editable,
    # so let's define an empty array to indicate no uneditable columns.
    ALL_CELLS_EDITABLE = []

    # -------------------------------------------------------------------------
    # Instance fields
    # -------------------------------------------------------------------------
    def __init__(self,
                 t=UNDEFINED_FLOAT,
                 workMode=UNDEFINED_STRING,
                 mTot=UNDEFINED_FLOAT,
                 mExt=UNDEFINED_FLOAT,
                 mRest=UNDEFINED_FLOAT,
                 walkSpeed=UNDEFINED_FLOAT,
                 load=UNDEFINED_FLOAT,
                 terrain=UNDEFINED_FLOAT,
                 grade=UNDEFINED_FLOAT,
                 clothing=UNDEFINED_STRING,
                 fluidIntake=UNDEFINED_FLOAT):
        """
        Constructs an ActivityRecord. Mirrors the Java approach:

        - The Java code has multiple constructors:
          (1) no-arg constructor
          (2) a constructor with time, workMode, walkSpeed, load, etc.
          (3) a constructor with time, workMode='m', mTot, mExt, mRest, etc.
        - This Python version merges them into one __init__ with optional arguments.
        """

        self.t = t
        self.workMode = workMode
        self.mTot = mTot
        self.mExt = mExt
        self.mRest = mRest
        self.walkSpeed = walkSpeed
        self.load = load
        self.terrain = terrain
        self.grade = grade
        self.fluidIntake = fluidIntake
        self.clothing = clothing

    # -------------------------------------------------------------------------
    # copyOf
    # -------------------------------------------------------------------------
    def copyOf(self, that):
        """
        Makes this ActivityRecord a copy of 'that' ActivityRecord.
        Mirrors: public void copyOf(ActivityRecord that)
        """
        self.t = that.t
        self.walkSpeed = that.walkSpeed

        if that.clothing != UNDEFINED_STRING:
            self.clothing = str(that.clothing)

        self.fluidIntake = that.fluidIntake

        if that.workMode != UNDEFINED_STRING:
            self.workMode = str(that.workMode)
            if that.workMode == "m":
                self.mTot  = that.mTot
                self.mExt  = that.mExt
                self.mRest = that.mRest
                return

        self.load    = that.load
        self.terrain = that.terrain
        self.grade   = that.grade

    # -------------------------------------------------------------------------
    # reset
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Resets instance to default values (undefined).
        Mirrors: public void reset()
        """
        self.t         = UNDEFINED_FLOAT
        self.workMode  = UNDEFINED_STRING
        self.mTot      = UNDEFINED_FLOAT
        self.mExt      = UNDEFINED_FLOAT
        self.mRest     = UNDEFINED_FLOAT
        self.walkSpeed = UNDEFINED_FLOAT
        self.load      = UNDEFINED_FLOAT
        self.terrain   = UNDEFINED_FLOAT
        self.grade     = UNDEFINED_FLOAT
        self.clothing  = UNDEFINED_STRING
        self.fluidIntake = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_t(self):
        """Gets the time [min]."""
        return self.t

    def getWorkMode(self):
        """Gets the work mode (e.g. 'r', 'f', 'm')."""
        return self.workMode

    def getMtot(self):
        """Gets the total metabolic rate [W]."""
        return self.mTot

    def getMext(self):
        """Gets the external work [W]."""
        return self.mExt

    def getMrest(self):
        """Gets the resting metabolic rate [W]."""
        return self.mRest

    def getWalkSpeed(self):
        """Gets the walk speed [m/s]."""
        return self.walkSpeed

    def getLoad(self):
        """Gets the load weight [kg]."""
        return self.load

    def getTerrain(self):
        """Gets the terrain factor (dimensionless)."""
        return self.terrain

    def getGrade(self):
        """Gets the terrain grade [%]."""
        return self.grade

    def getClothing(self):
        """Gets the clothing type (e.g. 'WBDU')."""
        return self.clothing

    def getFluidIntake(self):
        """Gets the fluid intake [gm/min]."""
        return self.fluidIntake

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def set_t(self, t):
        """Sets the time [min]."""
        self.t = t

    def setWorkMode(self, workMode):
        """Sets the work mode (e.g. 'r', 'f', 'm')."""
        self.workMode = workMode

    def setMtot(self, mtot):
        """Sets the total metabolic rate [W]."""
        self.mTot = mtot

    def setMext(self, mExt):
        """Sets the external work [W]."""
        self.mExt = mExt

    def setMrest(self, mRest):
        """Sets the resting metabolic rate [W]."""
        self.mRest = mRest

    def setWalkSpeed(self, walkSpeed):
        """Sets the walk speed [m/s]."""
        self.walkSpeed = walkSpeed

    def setLoad(self, load):
        """Sets the load weight [kg]."""
        self.load = load

    def setTerrain(self, terrain):
        """Sets the terrain factor (dimensionless)."""
        self.terrain = terrain

    def setGrade(self, grade):
        """Sets the terrain grade [%]."""
        self.grade = grade

    def setClothing(self, clothing):
        """Sets the clothing type."""
        self.clothing = clothing

    def setFluidIntake(self, fluidIntake):
        """Sets the fluid intake [gm/min]."""
        self.fluidIntake = fluidIntake

    # -------------------------------------------------------------------------
    # toString logic
    # -------------------------------------------------------------------------
    def __str__(self):
        """
        Builds a comma-separated string that consists of all data in this class.
        If the work mode is 'm', we call buildString() helper for that.
        Mirrors: public String toString()
        """
        if (self.workMode != UNDEFINED_STRING) and (self.workMode == "m"):
            return self.buildString()

        # We replicate the Java logic for 'r' or 'f' modes:
        if self.t != UNDEFINED_FLOAT:
            line = str(self.t)
        else:
            line = COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.workMode != UNDEFINED_STRING:
            line += COMMA_STRING + self.workMode
        else:
            line += COMMA_STRING + EMPTY_STRING

        if self.walkSpeed != UNDEFINED_FLOAT:
            # convert m/s to miles/hour
            mph = self.walkSpeed * SCENARIO_M_PER_SEC_TO_MILES_PER_HR
            line += COMMA_STRING + str(mph)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.load != UNDEFINED_FLOAT:
            lbs = self.load * SCENARIO_KG_TO_LBS
            line += COMMA_STRING + str(lbs)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.terrain != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.terrain)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.grade != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.grade)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.clothing != UNDEFINED_STRING:
            line += COMMA_STRING + self.clothing
        else:
            line += COMMA_STRING + EMPTY_STRING

        if self.fluidIntake != UNDEFINED_FLOAT:
            # gm/min -> l/hr
            val = self.fluidIntake * GM_PER_MIN_TO_L_PER_HR
            line += COMMA_STRING + str(val)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        return line

    def buildString(self):
        """
        A helper to build the string when work mode is 'm'.
        Mirrors: private String buildString() in Java.
        """
        if self.t != UNDEFINED_FLOAT:
            line = str(self.t)
        else:
            line = COMMA_STRING + str(UNDEFINED_FLOAT)

        line += COMMA_STRING + str(self.workMode)

        if self.mTot != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.mTot)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.mExt != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.mExt)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.mRest != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.mRest)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.walkSpeed != UNDEFINED_FLOAT:
            mph = self.walkSpeed * SCENARIO_M_PER_SEC_TO_MILES_PER_HR
            line += COMMA_STRING + str(mph)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.clothing != UNDEFINED_STRING:
            line += COMMA_STRING + self.clothing
        else:
            line += COMMA_STRING + EMPTY_STRING

        if self.fluidIntake != UNDEFINED_FLOAT:
            val = self.fluidIntake * GM_PER_MIN_TO_L_PER_HR
            line += COMMA_STRING + str(val)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        return line

    # -------------------------------------------------------------------------
    # computeMissingValues
    # -------------------------------------------------------------------------
    def computeMissingValues(self):
        """
        Computes class missing data, normally setting them to a standard value
        if they are undefined. Mirrors: public void computeMissingValues().
        """
        if self.workMode == UNDEFINED_STRING:
            self.workMode = self.STANDARD_WORKMODE

        if self.walkSpeed == UNDEFINED_FLOAT:
            self.walkSpeed = self.STANDARD_WALKSPEED

        if self.load == UNDEFINED_FLOAT:
            self.load = self.STANDARD_LOAD

        if self.terrain == UNDEFINED_FLOAT:
            self.terrain = self.STANDARD_TERRAIN

        if self.grade == UNDEFINED_FLOAT:
            self.grade = self.STANDARD_GRADE

        if self.fluidIntake == UNDEFINED_FLOAT:
            self.fluidIntake = self.STANDARD_FLUID_INTAKE

    # -------------------------------------------------------------------------
    # getColumnCount / getColumnIdentifiers / getLongValues / getToolTips
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
        """Gets an array of type placeholders for each column."""
        return cls.longValues

    @classmethod
    def getToolTips(cls):
        """Gets the tool tips for each column."""
        return cls.toolTips

    # -------------------------------------------------------------------------
    # getUnEditableOrdinals
    # -------------------------------------------------------------------------
    @classmethod
    def getUnEditableOrdinals(cls):
        """
        Gets the uneditable ordinals in the table logic. The Java code returns
        'ALL_CELLS_EDITABLE' but we have that as an empty list if all are editable.
        """
        return cls.ALL_CELLS_EDITABLE

    # -------------------------------------------------------------------------
    # saveValue / getValue
    # -------------------------------------------------------------------------
    def saveValue(self, index, value):
        """
        Saves the specified value at the given index, mirroring the Java code.
        Mirrors: public void saveValue(int index, Object value)
        """
        # If 'value' is an empty string, treat it as UNDEFINED_STRING
        if isinstance(value, str) and value == "":
            value = UNDEFINED_STRING

        if index == self.TIME:
            self.set_t(float(value))
        elif index == self.WORK_MODE:
            self.setWorkMode(str(value))
        elif index == self.MTOT:
            self.setMtot(float(value))
        elif index == self.MEXT:
            self.setMext(float(value))
        elif index == self.MREST:
            self.setMrest(float(value))
        elif index == self.WALK_SPEED:
            self.setWalkSpeed(float(value))
        elif index == self.LOAD:
            self.setLoad(float(value))
        elif index == self.TERRAIN:
            self.setTerrain(float(value))
        elif index == self.GRADE:
            self.setGrade(float(value))
        elif index == self.CLOTHING:
            self.setClothing(str(value))
        elif index == self.FLUID_INTAKE:
            self.setFluidIntake(float(value))

    def getValue(self, index):
        """
        Gets the value at the given column index, mirroring the Java code.
        Mirrors: public Object getValue(int index)
        """
        if index == self.TIME:
            return float(self.t)
        elif index == self.WORK_MODE:
            return self.workMode if self.workMode != None else EMPTY_STRING
        elif index == self.MTOT:
            return float(self.mTot)
        elif index == self.MEXT:
            return float(self.mExt)
        elif index == self.MREST:
            return float(self.mRest)
        elif index == self.WALK_SPEED:
            return float(self.walkSpeed)
        elif index == self.LOAD:
            return float(self.load)
        elif index == self.TERRAIN:
            return float(self.terrain)
        elif index == self.GRADE:
            return float(self.grade)
        elif index == self.CLOTHING:
            return self.clothing if self.clothing != None else EMPTY_STRING
        elif index == self.FLUID_INTAKE:
            return float(self.fluidIntake)
        return None


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class:
#
#   from scenario_python_2.activity_record import ActivityRecord
#
#   ar = ActivityRecord()
#   ar.set_t(10.0)            # 10 min
#   ar.setWorkMode('f')       # free walking
#   ar.setWalkSpeed(1.5)      # 1.5 m/s
#   ar.computeMissingValues() # fill in any standard defaults if undefined
#   print(ar)
#
# =============================================================================
