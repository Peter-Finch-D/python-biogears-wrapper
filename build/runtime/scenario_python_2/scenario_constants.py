# =============================================================================
# scenario_constants.py
#
# This Python module is a direct translation of the Java interface
# "ScenarioConstants" from the provided code. All constants are defined at the
# module level in Python. Where the original Java code uses 'public interface'
# and fields, we replicate them here as module-level constants. This is to keep
# a one-to-one correspondence with the original Java constants.
#
# IMPORTANT:
#   - Like in Java, these constants must be imported or referenced by other
#     modules (such as scenario_model.py).
#   - The usage example at the bottom shows how to do a simple import in Python.
# =============================================================================

# -- space character --
SPACE = ' '
# -- comma character --
COMMA = ','

# -- underscore character --
UNDEFINED_CHAR = '_'
# -- undefined int --
UNDEFINED_INT = -999
# -- undefined float --
UNDEFINED_FLOAT = -999.0
# -- undefined string --
UNDEFINED_STRING = None

# -- null string --
NULL_STRING = "<null>"
# -- empty string --
EMPTY_STRING = ""
# -- comma string --
COMMA_STRING = ","
# -- space string --
SPACE_STRING = " "

# -----------------------------------------------------------------------------
# Java's Math constants in Python
# -----------------------------------------------------------------------------
# Note: Python already has math.pi and math.e, but we define these for direct
# compatibility with the Java code.
import math

PI = math.pi
E = math.e

# -----------------------------------------------------------------------------
# Conversion factors
# -----------------------------------------------------------------------------
FT_TO_M = 0.3048
M_TO_FT = 1 / FT_TO_M
SCENARIO_KG_TO_LBS = 2.2046
SCENARIO_LBS_TO_KG = 1 / SCENARIO_KG_TO_LBS
SCENARIO_MILES_PER_HR_TO_M_PER_SEC = 0.44704
SCENARIO_M_PER_SEC_TO_MILES_PER_HR = 1 / SCENARIO_MILES_PER_HR_TO_M_PER_SEC
L_PER_HR_TO_GM_PER_MIN = 1000.0 / 60.0
GM_PER_MIN_TO_L_PER_HR = 1 / L_PER_HR_TO_GM_PER_MIN

# -----------------------------------------------------------------------------
# Acclimation and dehydration indices
# -----------------------------------------------------------------------------
NO_ACCLIMATION = 0
PART_ACCLIMATION = 1
FULL_ACCLIMATION = 2
DEHYD_NORMAL = 0
DEHYD_MODERATE = 1
DEHYD_SEVERE = 2

# -----------------------------------------------------------------------------
# Additional constants from SCENARIO.BAS
# -----------------------------------------------------------------------------
Lewis = 2.2
Boltz = 5.67e-08
SArad = 0.72
HtVap = 40.8
KClo  = 0.155

# Specific Heats (W-min/(kg-C))
SpHtCr = 51
SpHtMu = 63
SpHtFat = 42
SpHtSk = 63
SpHtBl = 62

fBWnfatCr  = 0.37
fBWnfatMu  = 0.54
fBWnfatVsk = 0.01
fBWnfatSk  = 0.06
fBWnfatRa  = 0.02

fMrstCr  = 0.824
fMrstMu  = 0.123
fMrstFat = 0.04
fMrstVsk = 0.002
fMrstSk  = 0.012

fCOrstCr  = 0.916
fCOrstMu  = 0.0492
fCOrstFat = 0.012
fCOrstVsk = 0.0228

Dfat  = 0.9
Dnfat = 1.1

# specific conductivities
kcr  = 54.29
kmu  = 41.76
kfat = 15.87
kvsk = 41.76
ksk  = 20.88