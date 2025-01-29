# =============================================================================
# inputs2.py
#
# This Python module is a direct translation of the Java class "Inputs2"
# from the provided code. It stores scenario metabolic and activity inputs.
# All functionality is intended to mirror the original Java version as closely
# as possible, line-by-line, preserving method names, logic, and comments.
#
# IMPORTANT:
#   - This code depends on "scenario_constants.py" for the constants
#     UNDEFINED_FLOAT and UNDEFINED_STRING (and presumably more).
#   - It also references "SubjectRecord" and "ActivityRecord" standard values
#     (e.g., STANDARD_WEIGHT, STANDARD_WORKMODE). In Python, you must define
#     or mock them in corresponding modules or handle them as needed in your
#     environment.
# =============================================================================

from scenario_constants import (
    UNDEFINED_FLOAT, UNDEFINED_STRING
)
from subject_record import SubjectRecord
from activity_record import ActivityRecord

class Inputs2:
    """
    A container for scenario metabolic and activity inputs.
    Translated from 'Inputs2.java'.
    """

    def __init__(self):
        """
        Default constructor. Sets all values to UNDEFINED_FLOAT or UNDEFINED_STRING.
        Mirrors: public Inputs2().
        """
        # metabolic rates
        self.Mtot   = UNDEFINED_FLOAT  # Total metabolic rate [W]
        self.Mrst   = UNDEFINED_FLOAT  # Resting metabolic rate [W]
        self.Mext   = UNDEFINED_FLOAT  # External work rate [W]
        self.Mwork  = UNDEFINED_FLOAT  # Work rate [W]
        self.PctEff = UNDEFINED_FLOAT  # % efficiency used to estimate external work

        # activity
        self.workMode   = UNDEFINED_STRING  # e.g. "r", "f", etc.
        self.Vmove      = UNDEFINED_FLOAT   # walk speed [m/s]
        self.load       = UNDEFINED_FLOAT   # subject load [kg]
        self.terrain    = UNDEFINED_FLOAT   # terrain factor (dimensionless)
        self.grade      = UNDEFINED_FLOAT   # terrain grade (dimensionless, e.g. 0.1 for 10%)
        self.fluidIntake = UNDEFINED_FLOAT  # fluid intake [gm/min]

    def copyOf(self, that):
        """
        Makes this Inputs2 a copy of that Inputs2.
        Mirrors: public void copyOf(Inputs2 that)
        """
        self.Mtot   = that.Mtot
        self.Mrst   = that.Mrst
        self.Mext   = that.Mext
        self.Mwork  = that.Mwork
        self.PctEff = that.PctEff

        # Java does new String(that.workMode.toString())
        self.workMode = str(that.workMode)

        self.Vmove       = that.Vmove
        self.load        = that.load
        self.terrain     = that.terrain
        self.grade       = that.grade
        self.fluidIntake = that.fluidIntake

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def getMtot(self):
        """Returns the total metabolic rate [W]."""
        return self.Mtot

    def getMrst(self):
        """Returns the resting metabolic rate [W]."""
        return self.Mrst

    def getMext(self):
        """Returns the external work rate [W]."""
        return self.Mext

    def getMwork(self):
        """Returns the work rate [W]."""
        return self.Mwork

    def getPctEff(self):
        """Returns the effectiveness used to estimate the external work [%]."""
        return self.PctEff

    def getWorkMode(self):
        """Returns the work mode (a mnemonic character, e.g. 'r', 'f')."""
        return self.workMode

    def getVmove(self):
        """Returns the walk speed [m/s]."""
        return self.Vmove

    def getLoad(self):
        """Returns the subject's load [kg]."""
        return self.load

    def getTerrain(self):
        """Returns the terrain factor (dimensionless)."""
        return self.terrain

    def getGrade(self):
        """Returns the terrain grade (dimensionless, e.g. 0.1 for 10%)."""
        return self.grade

    def getFluidIntake(self):
        """Returns the fluid intake [gm/min]."""
        return self.fluidIntake

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setMtot(self, Mtot):
        """Sets the total metabolic rate [W]."""
        self.Mtot = Mtot

    def setMrst(self, Mrst):
        """Sets the resting metabolic rate [W]."""
        self.Mrst = Mrst

    def setMext(self, Mext):
        """Sets the external work rate [W]."""
        self.Mext = Mext

    def setMwork(self, Mwork):
        """Sets the work rate [W]."""
        self.Mwork = Mwork

    def setPctEff(self, PctEff):
        """Sets the effectiveness used to compute external work [%]."""
        self.PctEff = PctEff

    def setWorkMode(self, workMode):
        """Sets the work mode (e.g. 'r' or 'f')."""
        self.workMode = workMode

    def setVmove(self, Vmove):
        """Sets the walk speed [m/s]."""
        self.Vmove = Vmove

    def setLoad(self, load):
        """Sets subject's load [kg]."""
        self.load = load

    def setGrade(self, grade):
        """Sets the terrain grade (dimensionless)."""
        self.grade = grade

    def setTerrain(self, terrain):
        """Sets the terrain factor (dimensionless)."""
        self.terrain = terrain

    def setFluidIntake(self, fluidIntake):
        """Sets the fluid intake [gm/min]."""
        self.fluidIntake = fluidIntake

    # -------------------------------------------------------------------------
    # Resetting methods
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Helper function to set all Inputs2 data to 'undefined' values.
        Mirrors: public void reset()
        """
        self.resetMetab()
        self.resetActivity()

    def resetMetab(self):
        """
        Sets all metabolic variables to 'undefined' values.
        Mirrors: public void resetMetab()
        """
        self.Mtot   = UNDEFINED_FLOAT
        self.Mrst   = UNDEFINED_FLOAT
        self.Mext   = UNDEFINED_FLOAT
        self.Mwork  = UNDEFINED_FLOAT
        self.PctEff = UNDEFINED_FLOAT

    def resetActivity(self):
        """
        Sets all activity variables to 'undefined' values.
        Mirrors: public void resetActivity()
        """
        self.workMode   = UNDEFINED_STRING
        self.Vmove      = UNDEFINED_FLOAT
        self.load       = UNDEFINED_FLOAT
        self.terrain    = UNDEFINED_FLOAT
        self.grade      = UNDEFINED_FLOAT
        self.fluidIntake= UNDEFINED_FLOAT

    # -------------------------------------------------------------------------
    # Standard values methods
    # -------------------------------------------------------------------------
    def setStandardValues(self):
        """
        Helper function to set all Inputs2 data to "standard" values.
        Mirrors: public void setStandardValues()
        """
        self.setStandardMetab()
        self.setStandardActivity()

    def setStandardMetab(self):
        """
        Sets all metabolic variables to "standard" values.
        Mirrors: public void setStandardMetab()
        """
        # This references SubjectRecord.STANDARD_WEIGHT, etc.
        # We'll do a direct import if they exist, or placeholders:
        self.Mtot = 1.5 * SubjectRecord.STANDARD_WEIGHT
        self.Mrst = 1.5 * SubjectRecord.STANDARD_WEIGHT
        self.Mext = 0.0  # max ~ 25% ?

    def setStandardActivity(self):
        """
        Sets all activity variables to "standard" values.
        Mirrors: public void setStandardActivity()
        """
        self.workMode   = ActivityRecord.STANDARD_WORKMODE
        self.Vmove      = ActivityRecord.STANDARD_WALKSPEED
        self.load       = ActivityRecord.STANDARD_LOAD
        self.terrain    = ActivityRecord.STANDARD_TERRAIN
        self.grade      = ActivityRecord.STANDARD_GRADE
        self.fluidIntake= ActivityRecord.STANDARD_FLUID_INTAKE

    # -------------------------------------------------------------------------
    # Compute missing values
    # -------------------------------------------------------------------------
    def computeMissingValues(self, BW):
        """
        Computes missing/undefined values. 
        'BW' is the subject weight [kg].
        Mirrors: public void computeMissingValues(float BW)
        """
        # Mrst -> Mwork -> Mtot -> Mext

        # 1) if Mrst is undefined, compute Mrst
        if self.Mrst == UNDEFINED_FLOAT:
            self.computeMrst(BW)

        # 2) Mwork is always undefined in the Java code, so compute it
        if self.Mwork == UNDEFINED_FLOAT:
            self.computeMwork(BW)

        # now Mtot = Mrst + Mwork
        self.Mtot = self.Mrst + self.Mwork

        # 3) if Mext is undefined, compute Mext
        if self.Mext == UNDEFINED_FLOAT:
            self.computeMext(BW)

        # hack from Java to prevent Mnet < Mrst:
        # Mnet = Mtot - Mext
        if self.Mext > self.Mwork:
            self.Mext = self.Mwork

    def computeMrst(self, BW):
        """
        Computes the resting metabolic rate.
        Mirrors private void computeMrst(float BW).
        """
        self.Mrst = 1.5 * BW  # empirical estimate

    def computeMwork(self, BW):
        """
        Computes the working metabolic rate.
        Mirrors private void computeMwork(float BW).
        """
        # if Mtot != UNDEFINED_FLOAT, then Mwork = Mtot - Mrst
        if self.Mtot != UNDEFINED_FLOAT:
            self.Mwork = self.Mtot - self.Mrst
        elif self.workMode == "f":
            # 'f' for free-walking
            # logic from Java:
            # Mwork = 2*(BW+load)*((load/BW)^2)
            #   + terrain*(BW+load)*(1.5*(Vmove^2)+0.35*Vmove*grade)
            val1 = 2.0*(BW + self.load)*( (self.load/BW)**2 )
            val2 = self.terrain*(BW + self.load)*(
                     1.5*(self.Vmove**2) + 0.35*self.Vmove*self.grade
                   )
            self.Mwork = val1 + val2

            if self.grade < 0:
                # use same code from Java
                cf = ( self.terrain * ((self.grade*(BW+self.load)*self.Vmove)/3.5
                      - ((BW+self.load)*((self.grade+6)**2)/BW)
                      + (25.0 - (self.Vmove**2))) )
                self.Mwork -= cf
        else:
            # else assume 0
            self.Mwork = 0.0

    def computeMext(self, BW):
        """
        Computes the external work.
        Mirrors private void computeMext(float BW).
        """
        # only compute if Mtot>Mrst (i.e., working)
        if self.Mtot > self.Mrst:
            if self.PctEff != UNDEFINED_FLOAT:
                self.Mext = self.Mtot * self.PctEff / 100.0
            elif self.workMode == "f":
                # free-walking
                self.Mext = 0.098 * self.grade * (BW + self.load) * self.Vmove
            else:
                self.Mext = self.Mtot * 0.2  # default to 20%
        else:
            self.Mext = 0.0

    def __str__(self):
        """
        Reports variable values. Mirrors the Java toString().
        """
        out = (f"Mtot:{self.Mtot} "
               f"Mrst:{self.Mrst} "
               f"Mext:{self.Mext} "
               f"Mwork:{self.Mwork} "
               f"PctEff:{self.PctEff} "
               f"workMode:{self.workMode} "
               f"Vmove:{self.Vmove} "
               f"load:{self.load} "
               f"terrain:{self.terrain} "
               f"grade:{self.grade} "
               f"fluidIntake:{self.fluidIntake}")
        return out