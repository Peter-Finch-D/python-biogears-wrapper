"""
utils.py

A Python module replicating various static utility functions from Utils.java.
"""

import math
import os
import sys
import datetime
from typing import Optional

# Some Java-like methods from Utils.java that are commonly used in SCENARIO:

def SatVP(Temp: float) -> float:
    """
    Returns saturation vapor pressure [Torr] for temperature [C].
    This is the Java version's approach (slightly simplified).
    """
    Tk = 273.0 + Temp
    # Original logic from Scenario's Utils:
    # LPsat = 28.59051 - (8.2 * log10(Tk)) + (.00248*Tk) - (3142.3/Tk)
    # Psat(mbars) = 1000 * 10^(LPsat)
    # Then convert mbars -> Torr
    # For a simpler approach, use that logic or replicate exactly:
    LPsat = 28.59051 - 8.2 * math.log10(Tk) + 0.00248 * Tk - (3142.3 / Tk)
    Psat_mbar = 1000.0 * math.pow(10.0, LPsat)
    Psat_torr = Psat_mbar * 0.75
    return Psat_torr

def LagIt(OldVal: float, NewVal: float, time: float, HalfTime: float) -> float:
    """
    The classic exponential 'LagIt' from the Java code.
    LagVal = OldVal + (NewVal - OldVal)*(1 - e^(-0.693 * time / HalfTime))
    """
    if HalfTime <= 0.0:
        return NewVal
    # 0.693 is ln(2). Using the Java's exact expression:
    val = OldVal + (NewVal - OldVal) * (1.0 - math.exp(-0.693 * time / HalfTime))
    return val

def DuBois(BW: float, HT: float) -> float:
    """
    Classic DuBois formula for body surface area:
    DuBois = 0.202 * BW^0.425 * (HT/100)^0.725
    BW in kg, HT in cm -> result in m^2
    """
    return 0.202 * (BW**0.425) * ((HT / 100.0)**0.725)

# Additional placeholder methods from Utils.java
# (If you do not need them, you can safely remove them.)
def readProfileString(file, section, key, err_msg):
    """
    Java readProfileString(...) stub. 
    Returns err_msg if the file or key is not found,
    otherwise returns the line after '='.
    """
    # This is just a skeleton to mimic Java behavior.
    if not os.path.exists(file):
        return err_msg

    s = err_msg
    with open(file, "r") as f:
        lines = f.readlines()

    sectionFound = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # If we encounter a new section
        if line.startswith("[") and sectionFound:
            # we've left the old section
            break
        if line.startswith(section):
            # We are in the right section now
            sectionFound = True
            continue
        # If we are in the section, look for key
        if sectionFound and line.startswith(key):
            idx = line.find('=')
            if idx == -1:
                break
            # parse the value
            candidate_key = line[:idx].strip()
            if candidate_key == key:
                s = line[idx+1:].strip()
                break
    return s

def writeProfileString(file, section, key, value, err_msg):
    """
    Java writeProfileString(...) stub. Overwrites or appends key=value in a section.
    If the file or section not found, return err_msg. 
    """
    return "key written"

def removeProfileString(file, section, key, err_msg):
    """
    Java removeProfileString(...) stub. 
    """
    return "key removed"

def scenarioTimeIs() -> str:
    """ 
    Mimics scenarioTimeIs() from Java Utils: returns a short date + medium time. 
    """
    now = datetime.datetime.now()
    return now.strftime("%m/%d/%y, %I:%M:%S %p")
