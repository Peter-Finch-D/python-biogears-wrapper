# =============================================================================
# circadian_table.py
#
# This Python module is a direct translation of the Java class "CircadianTable"
# from the provided code. All functionality is intended to mirror the original
# Java version as closely as possible, line-by-line, preserving method names,
# logic, and comments.
#
# In Java, the arrays `time[]` and `temp[]` and the integer `size` were static,
# meaning they were shared by all instances. Here we mimic that by storing them
# as class-level attributes in Python. The `iLast` is an instance-level attribute
# because in Java it was a non-static field (i.e., an instance variable).
#
# USAGE NOTE:
#   - `setTime()`, `setTemp()`, and `setSize()` are class methods. They change
#     the class-level arrays `time`, `temp`, and the integer `size` for *all*
#     instances of CircadianTable. This is the same behavior as in the Java code,
#     where these fields were static.
# =============================================================================

class CircadianTable:
    """
    Holds circadian data, mirroring the Java 'CircadianTable' class.
    """

    # class-level constants (mimicking Java's 'public static final')
    MAX_SIZE = 100
    TIME = 1
    TEMP = 2
    NUM_FIELDS = 2

    # class-level arrays and size (mimicking Java's static float[] time, temp; static int size)
    time = [0.0] * MAX_SIZE
    temp = [0.0] * MAX_SIZE
    size = 0

    def __init__(self):
        """
        Default constructor. 
        'iLast' is an instance variable: index of the upper time bound of the last interpolation.
        Mirrors the Java code where iLast is non-static.
        """
        self.iLast = 1

    @classmethod
    def setTime(cls, time_value, index):
        """
        Sets the time at the specified index in the class-level 'time' array.
        Mirrors: public static void setTime(float time1, int index)
        """
        cls.time[index] = time_value

    @classmethod
    def setTemp(cls, temp_value, index):
        """
        Sets the temperature at the specified index in the class-level 'temp' array.
        Mirrors: public static void setTemp(float temp1, int index)
        """
        cls.temp[index] = temp_value

    @classmethod
    def setSize(cls, size_value):
        """
        Sets the class-level 'size'.
        Mirrors: public static void setSize(int size1)
        """
        cls.size = size_value

    def interpTemp(self, t):
        """
        Finds the circadian temperature at the given time t (0-24 hours).
        For speed, the routine remembers the last bracket used via 'iLast'.
        For random jumps, it searches bottom bracket or top bracket.
        Mirrors: public float interpTemp(float t)
        """
        i = self.iLast

        # find bottom bracket
        while t < self.time[i - 1]:
            i -= 1

        # find upper bracket
        while t > self.time[i]:
            i += 1

        # linear interpolation
        k = (t - self.time[i - 1]) / (self.time[i] - self.time[i - 1])
        interp_temp = self.temp[i - 1] + k * (self.temp[i] - self.temp[i - 1])

        # remember this bracket
        self.iLast = i

        return interp_temp


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import this class in scenario_model.py:
#
#   from scenario_python_2.circadian_table import CircadianTable
#
#   # Setting up static data:
#   CircadianTable.setSize(3)
#   CircadianTable.setTime(0.0, 0)
#   CircadianTable.setTemp(37.0, 0)
#   CircadianTable.setTime(12.0, 1)
#   CircadianTable.setTemp(37.5, 1)
#   CircadianTable.setTime(24.0, 2)
#   CircadianTable.setTemp(37.0, 2)
#
#   # Then in usage:
#   circadian = CircadianTable()
#   temp_at_10 = circadian.interpTemp(10.0)
#   print("Temp at 10 hours:", temp_at_10)
#
# =============================================================================
