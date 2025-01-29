# =============================================================================
# clothing_record.py
#
# This Python module is a direct translation of the Java class "ClothingRecord"
# from the provided code. It holds attributes for a type of clothing, including
# certain clo values and computed factors Iclo, Im. All functionality is
# intended to mirror the original Java version as closely as possible,
# line-by-line, preserving method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for constants such as
#     UNDEFINED_FLOAT, UNDEFINED_STRING, EMPTY_STRING, COMMA_STRING, etc.
#   - If you have a different approach to how constants are defined, adjust
#     accordingly.
# =============================================================================

from scenario_constants import (
    UNDEFINED_FLOAT, UNDEFINED_STRING,
    EMPTY_STRING, COMMA_STRING
)

class ClothingRecord:
    """
    Holds attributes for a type of clothing, mirroring 'ClothingRecord.java'.
    """

    # -------------------------------------------------------------------------
    # Public static final constants from Java
    # -------------------------------------------------------------------------
    STANDARD_ICLO = 0.16    # (shorts)
    STANDARD_IM   = 0.51    # (shorts)

    # column indexes
    NAME  = 1
    CLO_A = 2
    CLO_B = 3
    CLO_C = 4
    CLO_D = 5

    NUM_FIELDS = 5

    colIdentifiers = ["Name ",
                      " clo_a ",
                      " clo_b ",
                      " clo_c ",
                      " clo_d "]

    longValues = [
        str("_WBDU_"),  # placeholder for Name
        float(2.222),
        float(2.222),
        float(2.222),
        float(2.222)
    ]

    toolTips = [
        "clothing parameters description",
        "clo_a",
        "clo_b",
        "clo_c",
        "clo_d"
    ]

    ALL_CELLS_EDITABLE = []  # indicates no columns are locked in the Java table model

    def __init__(self, name=UNDEFINED_STRING,
                 clo_a=UNDEFINED_FLOAT, clo_b=UNDEFINED_FLOAT,
                 clo_c=UNDEFINED_FLOAT, clo_d=UNDEFINED_FLOAT):
        """
        Mirrors the Java constructors:
          - public ClothingRecord() {}
          - public ClothingRecord(String name, float clo_a, float clo_b, float clo_c, float clo_d)
        """
        self.name  = name
        self.clo_a = clo_a
        self.clo_b = clo_b
        self.clo_c = clo_c
        self.clo_d = clo_d

        # computed values
        self.Iclo = UNDEFINED_FLOAT
        self.Im   = UNDEFINED_FLOAT

    def copyOf(self, clothingRecord):
        """
        Makes a copy of the specified ClothingRecord.
        Mirrors: public void copyOf(ClothingRecord clothingRecord)
        """
        self.name  = str(clothingRecord.getName())
        self.clo_a = clothingRecord.get_a()
        self.clo_b = clothingRecord.get_b()
        self.clo_c = clothingRecord.get_c()
        self.clo_d = clothingRecord.get_d()
        self.Iclo  = clothingRecord.get_Iclo()
        self.Im    = clothingRecord.get_Im()

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getName(self):
        """Gets the name of this clothing record."""
        return self.name

    def get_a(self):
        """Gets clo_a."""
        return self.clo_a

    def get_b(self):
        """Gets clo_b."""
        return self.clo_b

    def get_c(self):
        """Gets clo_c."""
        return self.clo_c

    def get_d(self):
        """Gets clo_d."""
        return self.clo_d

    def get_Iclo(self):
        """Gets the clothing factor Iclo."""
        return self.Iclo

    def get_Im(self):
        """Gets the clothing factor Im."""
        return self.Im

    # -------------------------------------------------------------------------
    # Resets for Iclo / Im
    # -------------------------------------------------------------------------
    def resetImIclo(self):
        """
        Sets Iclo and Im to UNDEFINED_FLOAT.
        Mirrors: public void resetImIclo()
        """
        self.Iclo = UNDEFINED_FLOAT
        self.Im   = UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def set_a(self, clo_a):
        """Sets clo_a."""
        self.clo_a = clo_a

    def set_b(self, clo_b):
        """Sets clo_b."""
        self.clo_b = clo_b

    def set_c(self, clo_c):
        """Sets clo_c."""
        self.clo_c = clo_c

    def set_d(self, clo_d):
        """Sets clo_d."""
        self.clo_d = clo_d

    def setIclo(self, Iclo):
        """Sets Iclo."""
        self.Iclo = Iclo

    def setIm(self, Im):
        """Sets Im."""
        self.Im = Im

    # -------------------------------------------------------------------------
    # Compute Iclo / Im
    # -------------------------------------------------------------------------
    def computeImIclo(self, Vair, Vmove):
        """
        Computes Im and Iclo based on the formula from the Java code.
        Mirrors: public void computeImIclo(float Vair, float Vmove)
        """
        # local placeholders
        Veff = 1.0
        Itot = 0.0
        Facl = 0.0
        ImOverClo = 0.0

        if self.clo_a != UNDEFINED_FLOAT and self.clo_b != UNDEFINED_FLOAT:
            Veff = Vair + Vmove
            Itot = self.clo_a * (Veff ** self.clo_b)
            Facl = 1.0 + 0.2 * self.clo_a
            # from code: Iclo = Itot - 1 / (.61 + 1.87 * sqrt(Veff)) / Facl;
            # we replicate that parentheses carefully
            self.Iclo = Itot - (1.0 / (0.61 + 1.87 * (Veff ** 0.5))) / Facl
        else:
            self.Iclo = self.STANDARD_ICLO

        if (self.clo_a != UNDEFINED_FLOAT and
            self.clo_c != UNDEFINED_FLOAT and
            self.clo_d != UNDEFINED_FLOAT):
            Veff = Vair + Vmove
            ImOverClo = self.clo_c * (Veff ** self.clo_d)
            Facl = 1.0 + 0.2 * self.clo_a
            Itot = self.Iclo + 1.0 / (0.61 + 1.87 * (Veff ** 0.5)) / Facl
            self.Im = ImOverClo * Itot
        else:
            self.Im = self.STANDARD_IM

    # -------------------------------------------------------------------------
    # toString
    # -------------------------------------------------------------------------
    def __str__(self):
        """
        Builds a comma-separated string that consists of all data in this class.
        Mirrors: public String toString().
        """
        if self.name != UNDEFINED_STRING:
            line = self.name
        else:
            from scenario_python_2.scenario_constants import NULL_STRING
            line = NULL_STRING

        from scenario_python_2.scenario_constants import UNDEFINED_FLOAT

        if self.clo_a != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.clo_a)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.clo_b != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.clo_b)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.clo_c != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.clo_c)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        if self.clo_d != UNDEFINED_FLOAT:
            line += COMMA_STRING + str(self.clo_d)
        else:
            line += COMMA_STRING + str(UNDEFINED_FLOAT)

        return line

    # -------------------------------------------------------------------------
    # Class-level table methods
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
        """Gets the column data types."""
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
        Saves the specified value at the given index. Mirrors:
        public void saveValue(int index, Object value).
        """
        if isinstance(value, str) and value == "":
            value = UNDEFINED_STRING

        if index == self.NAME:
            self.name = str(value)
        elif index == self.CLO_A:
            self.set_a(float(value))
        elif index == self.CLO_B:
            self.set_b(float(value))
        elif index == self.CLO_C:
            self.set_c(float(value))
        elif index == self.CLO_D:
            self.set_d(float(value))

    def getValue(self, index):
        """
        Gets the value at the given column index.
        Mirrors: public Object getValue(int index).
        """
        if index == self.NAME:
            return self.name if self.name else EMPTY_STRING
        elif index == self.CLO_A:
            return float(self.clo_a)
        elif index == self.CLO_B:
            return float(self.clo_b)
        elif index == self.CLO_C:
            return float(self.clo_c)
        elif index == self.CLO_D:
            return float(self.clo_d)
        return None
