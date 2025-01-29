# =============================================================================
# subject_record.py
#
# This Python module is a direct translation of the Java class "SubjectRecord"
# from the provided code. It holds attributes for a subject such as height,
# weight, age, body fat, ID, etc. All functionality is intended to mirror
# the original Java version as closely as possible, line-by-line, preserving
# method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for constants such as
#     UNDEFINED_FLOAT, UNDEFINED_INT, UNDEFINED_STRING, EMPTY_STRING,
#     M_TO_FT, etc.
#   - It also depends on a "Utils" module that provides Utils.DuBois(...) or
#     must replicate that logic. In the Java code, it references
#     mil.army.usariem.scenario.utils.Utils.
#   - The usage example at the bottom is abridged only; everything else is
#     full and unabridged.
# =============================================================================

from scenario_constants import (
    UNDEFINED_FLOAT, UNDEFINED_INT, UNDEFINED_STRING,
    EMPTY_STRING, COMMA_STRING,
    M_TO_FT
)
from utils import DuBois

class SubjectRecord:
    """
    Holds the subject's attributes such as height, weight, age, body fat, etc.
    Translated from 'SubjectRecord.java'.
    """

    # -------------------------------------------------------------------------
    # Public static final constants from Java
    # -------------------------------------------------------------------------
    STANDARD_WEIGHT  = 70.0    # kg
    STANDARD_HEIGHT  = 170.0   # cm
    STANDARD_AGE     = 23
    STANDARD_ACCL    = 0
    STANDARD_PCT_FAT = 14.0
    STANDARD_SA = DuBois(STANDARD_WEIGHT, STANDARD_HEIGHT)  # calls Utils.DuBois
    STANDARD_GENDER  = "m"
    STANDARD_PHASE   = UNDEFINED_STRING
    STANDARD_VO2MAX  = 14.0

    # ordinals for columns
    ID       = 1
    LASTNAME = 2
    FIRSTNAME= 3
    SSN      = 4
    DOB      = 5
    AGE      = 6
    HEIGHT   = 7
    WEIGHT   = 8
    BFAT     = 9

    # token identifiers
    ID_TOKEN       = ord('I')
    LASTNAME_TOKEN = ord('L')
    FIRSTNAME_TOKEN= ord('F')
    SSN_TOKEN      = ord('S')
    DOB_TOKEN      = ord('D')
    AGE_TOKEN      = ord('A')
    HEIGHT_M_TOKEN = ord('H')
    HEIGHT_FT_TOKEN= ord('h')
    WEIGHT_TOKEN   = ord('W')
    BFAT_TOKEN     = ord('B')

    METERS = 1
    FEET   = 2

    # column identifiers, data types, and tooltips
    colIdentifiers = [
        " ID ",
        " Last Name ",
        " First Name ",
        " SSN ",
        " DOB ",
        " Age ",
        " Height ",
        " Weight ",
        " Body Fat "
    ]
    longValues = [
        int(1000),
        str("Johnson"),
        str("Richard"),
        str("123-45-4321"),
        str("25-December-00"),
        int(100),
        float(2.222),
        float(2.222),
        float(2.222)
    ]
    toolTips = [
        "subject ID",
        "last name",
        "first name",
        "social security no.",
        "date of birth",
        "age",
        "height (cm)",
        "weight (kg)",
        "body fat (%)"
    ]

    ALL_CELLS_EDITABLE = []

    def __init__(self):
        """
        Constructs a SubjectRecord with no arguments. 
        Mirrors: public SubjectRecord().
        """
        self.lastname  = UNDEFINED_STRING
        self.firstname = UNDEFINED_STRING
        self.ssn       = UNDEFINED_STRING
        self.dob       = UNDEFINED_STRING
        self.id        = UNDEFINED_INT
        self.age       = UNDEFINED_INT
        self.height    = UNDEFINED_FLOAT  # [cm]
        self.weight    = UNDEFINED_FLOAT  # [kg]
        self.bfat      = UNDEFINED_FLOAT  # [%]
        self.height_flag = self.METERS

        self.gender    = UNDEFINED_STRING
        self.phase     = UNDEFINED_STRING
        self.vo2max    = UNDEFINED_FLOAT
        self.accl      = UNDEFINED_INT
        self.sa        = UNDEFINED_FLOAT

    def copyOf(self, subjectRecord):
        """
        Makes a copy of the specified subjectRecord.
        Mirrors: public void copyOf(SubjectRecord subjectRecord)
        """
        # If the string is not None, replicate it as new string
        val = subjectRecord.getLastname()
        self.lastname = str(val) if val is not None else None

        val = subjectRecord.getFirstname()
        self.firstname = str(val) if val is not None else None

        val = subjectRecord.getSSN()
        self.ssn = str(val) if val is not None else None

        val = subjectRecord.getDOB()
        self.dob = str(val) if val is not None else None

        self.id     = subjectRecord.getID()
        self.age    = subjectRecord.getAge()
        self.height = subjectRecord.getHeight()
        self.weight = subjectRecord.getWeight()
        self.bfat   = subjectRecord.getBfat()
        self.height_flag = subjectRecord.getHeightFlag()

        val = subjectRecord.getGender()
        self.gender = str(val) if val is not None else None

        val = subjectRecord.getPhase()
        self.phase = str(val) if val is not None else None

        self.sa     = subjectRecord.getSA()
        self.vo2max = subjectRecord.getVO2Max()
        self.accl   = subjectRecord.getAccl()

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getLastname(self):
        """Gets the last name."""
        return self.lastname

    def getFirstname(self):
        """Gets the first name."""
        return self.firstname

    def getSSN(self):
        """Gets the social security number."""
        return self.ssn

    def getDOB(self):
        """Gets the date of birth."""
        return self.dob

    def getID(self):
        """Gets the subject ID."""
        return self.id

    def getAge(self):
        """Gets the subject age [years]."""
        return self.age

    def getHeightFlag(self):
        """Gets the subject's height flag (meters or feet)."""
        return self.height_flag

    def getHeight(self):
        """Gets the subject's height [cm]."""
        return self.height

    def getWeight(self):
        """Gets the subject's weight [kg]."""
        return self.weight

    def getBfat(self):
        """Gets the subject's body fat [%]."""
        return self.bfat

    def getGender(self):
        """Gets the subject's gender ('m' or 'f', or undefined)."""
        return self.gender

    def getPhase(self):
        """Gets the subject's phase."""
        return self.phase

    def getSA(self):
        """Gets the subject's surface area [m^2]."""
        return self.sa

    def getVO2Max(self):
        """Gets the subject's VO2 max."""
        return self.vo2max

    def getAccl(self):
        """Gets the subject's acclimation value."""
        return self.accl

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Sets all class members to default 'undefined' values.
        Mirrors: public void reset()
        """
        self.lastname  = UNDEFINED_STRING
        self.firstname = UNDEFINED_STRING
        self.ssn       = UNDEFINED_STRING
        self.dob       = UNDEFINED_STRING
        self.id        = UNDEFINED_INT
        self.age       = UNDEFINED_INT
        self.height    = UNDEFINED_FLOAT
        self.weight    = UNDEFINED_FLOAT
        self.bfat      = UNDEFINED_FLOAT
        self.height_flag = self.METERS

        self.gender = UNDEFINED_STRING
        self.phase  = UNDEFINED_STRING
        self.vo2max = UNDEFINED_FLOAT
        self.accl   = UNDEFINED_INT
        self.sa     = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setLastname(self, lastname):
        """Sets the subject's last name."""
        self.lastname = lastname

    def setFirstname(self, firstname):
        """Sets the subject's first name."""
        self.firstname = firstname

    def setSSN(self, ssn):
        """Sets the subject's social security number."""
        self.ssn = ssn

    def setDOB(self, dob):
        """Sets the subject's date-of-birth."""
        self.dob = dob

    def setID(self, id_val):
        """Sets the subject's ID."""
        self.id = id_val

    def setAge(self, age_val):
        """Sets the subject's age [years]."""
        self.age = age_val

    def setHeight(self, height):
        """Sets the subject's height [cm]."""
        self.height = height

    def setWeight(self, weight):
        """Sets the subject's weight [kg]."""
        self.weight = weight

    def setBfat(self, bfat):
        """Sets the subject's body fat [%]."""
        self.bfat = bfat

    def setHeightFlag(self, flag):
        """Sets the subject's height flag (meter or feet)."""
        self.height_flag = flag

    def setSA(self, sa):
        """Sets the subject's surface area [m^2]."""
        self.sa = sa

    def setVO2Max(self, vo2max):
        """Sets the subject's VO2 max."""
        self.vo2max = vo2max

    def setAccl(self, accl):
        """Sets the subject's acclimation value."""
        self.accl = accl

    def setGender(self, gender):
        """Sets the subject's gender ('m' or 'f')."""
        self.gender = gender

    def setPhase(self, phase):
        """Sets the subject's phase."""
        self.phase = phase

    # -------------------------------------------------------------------------
    # toString
    # -------------------------------------------------------------------------
    def __str__(self):
        """
        Builds a comma-separated string of all data in this class, with tokens
        indicating the type (ID, LASTNAME, etc.). Mirrors: public String toString().
        """
        line = f"{chr(self.ID_TOKEN)}{self.id}"

        if self.lastname != UNDEFINED_STRING:
            s = f"{chr(self.LASTNAME_TOKEN)}{self.lastname}"
            line += COMMA_STRING + s

        if self.firstname != UNDEFINED_STRING:
            s = f"{chr(self.FIRSTNAME_TOKEN)}{self.firstname}"
            line += COMMA_STRING + s

        if self.ssn != UNDEFINED_STRING:
            s = f"{chr(self.SSN_TOKEN)}{self.ssn}"
            line += COMMA_STRING + s

        if self.dob != UNDEFINED_STRING:
            s = f"{chr(self.DOB_TOKEN)}{self.dob}"
            line += COMMA_STRING + s

        if self.age != UNDEFINED_INT:
            s = f"{chr(self.AGE_TOKEN)}{self.age}"
            line += COMMA_STRING + s

        if self.height != UNDEFINED_FLOAT:
            if self.height_flag == self.FEET:
                # convert cm -> m -> ft
                hgt = self.height * M_TO_FT / 100.0
                s = f"{chr(self.HEIGHT_FT_TOKEN)}{hgt}"
            else:
                # interpret as meters
                s = f"{chr(self.HEIGHT_M_TOKEN)}{(self.height / 100.0)}"
            line += COMMA_STRING + s

        if self.weight != UNDEFINED_FLOAT:
            s = f"{chr(self.WEIGHT_TOKEN)}{self.weight}"
            line += COMMA_STRING + s

        if self.bfat != UNDEFINED_FLOAT:
            s = f"{chr(self.BFAT_TOKEN)}{self.bfat}"
            line += COMMA_STRING + s

        return line

    # -------------------------------------------------------------------------
    # computeMissingValues
    # -------------------------------------------------------------------------
    def computeMissingValues(self):
        """
        Computes essential data that is still UNDEFINED.
        Mirrors: public void computeMissingValues().
        """
        if self.weight == UNDEFINED_FLOAT:
            self.weight = self.STANDARD_WEIGHT

        if self.height == UNDEFINED_FLOAT:
            self.height = self.STANDARD_HEIGHT

        if self.sa == UNDEFINED_FLOAT:
            self.sa = DuBois(self.weight, self.height)

        if self.age == UNDEFINED_FLOAT:
            self.age = self.STANDARD_AGE

        if self.gender == UNDEFINED_STRING:
            self.gender = self.STANDARD_GENDER

        if self.phase == UNDEFINED_STRING:
            self.phase = self.STANDARD_PHASE

        if self.bfat == UNDEFINED_FLOAT:
            self.bfat = self.STANDARD_PCT_FAT

        if self.vo2max == UNDEFINED_FLOAT:
            self.vo2max = self.STANDARD_VO2MAX

        if self.accl == UNDEFINED_INT:
            self.accl = self.STANDARD_ACCL

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
        """Gets the column data types placeholders."""
        return cls.longValues

    @classmethod
    def getToolTips(cls):
        """Gets the tool tips for each column."""
        return cls.toolTips

    # -------------------------------------------------------------------------
    # saveValue / getValue
    # -------------------------------------------------------------------------
    def saveValue(self, index, value):
        """
        Saves the specified value at the given index, mirroring the Java code.
        Mirrors: public void saveValue(int index, Object value)
        """
        if isinstance(value, str) and value == "":
            value = UNDEFINED_STRING

        if index == self.ID:
            self.setID(int(value))
        elif index == self.LASTNAME:
            self.setLastname(str(value))
        elif index == self.FIRSTNAME:
            self.setFirstname(str(value))
        elif index == self.SSN:
            self.setSSN(str(value))
        elif index == self.DOB:
            self.setDOB(str(value))
        elif index == self.HEIGHT:
            flt = float(value)
            self.setHeight(flt)
        elif index == self.WEIGHT:
            flt = float(value)
            self.setWeight(flt)
        elif index == self.BFAT:
            flt = float(value)
            self.setBfat(flt)

    def getValue(self, index):
        """
        Gets the value at the given column index.
        Mirrors: public Object getValue(int index)
        """
        if index == self.ID:
            return int(self.id)
        elif index == self.LASTNAME:
            return self.lastname if self.lastname else EMPTY_STRING
        elif index == self.FIRSTNAME:
            return self.firstname if self.firstname else EMPTY_STRING
        elif index == self.SSN:
            return self.ssn if self.ssn else EMPTY_STRING
        elif index == self.DOB:
            return self.dob if self.dob else EMPTY_STRING
        elif index == self.AGE:
            return int(self.age)
        elif index == self.HEIGHT:
            return float(self.height)
        elif index == self.WEIGHT:
            return float(self.weight)
        elif index == self.BFAT:
            return float(self.bfat)
        return None

    @classmethod
    def getUnEditableOrdinals(cls):
        """Gets the uneditable ordinals. By default, all cells are editable."""
        return cls.ALL_CELLS_EDITABLE
