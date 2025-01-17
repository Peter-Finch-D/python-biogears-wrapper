"""
scenario_constants.py

A Python module replicating the constants from ScenarioConstants.java
"""

import math

PI = math.pi

# Simple placeholders for undefined markers:
UNDEFINED_CHAR = '_'
UNDEFINED_INT = -999
UNDEFINED_FLOAT = -999.0
UNDEFINED_STRING = None

# Some string constants
NULL_STRING = "<null>"
EMPTY_STRING = ""
COMMA_STRING = ","
SPACE_STRING = " "

# Basic conversions
FT_TO_M = 0.3048
M_TO_FT = 1 / FT_TO_M
SCENARIO_KG_TO_LBS = 2.2046
SCENARIO_LBS_TO_KG = 1 / SCENARIO_KG_TO_LBS
SCENARIO_MILES_PER_HR_TO_M_PER_SEC = 0.44704
SCENARIO_M_PER_SEC_TO_MILES_PER_HR = 1 / SCENARIO_MILES_PER_HR_TO_M_PER_SEC
L_PER_HR_TO_GM_PER_MIN = 1000.0 / 60.0
GM_PER_MIN_TO_L_PER_HR = 1 / L_PER_HR_TO_GM_PER_MIN

# Acclimation and dehydration indices
NO_ACCLIMATION = 0
PART_ACCLIMATION = 1
FULL_ACCLIMATION = 2
DEHYD_NORMAL = 0
DEHYD_MODERATE = 1
DEHYD_SEVERE = 2

# Additional scenario-specific constants:
Lewis = 2.2
Boltz = 5.67e-8
SArad = 0.72
HtVap = 40.8        # W-min/g
KClo = 0.155        # m^2-C/W-Clo

# Specific Heats (W-min/(kg-C))
SpHtCr = 51.0
SpHtMu = 63.0
SpHtFat = 42.0
SpHtSk = 63.0
SpHtBl = 62.0

# Fraction of body weight (non-fat) assigned to compartments:
fBWnfatCr  = 0.37
fBWnfatMu  = 0.54
fBWnfatVsk = 0.01
fBWnfatSk  = 0.06
fBWnfatRa  = 0.02

# Distribution of resting metabolism:
fMrstCr  = 0.824
fMrstMu  = 0.123
fMrstFat = 0.04
fMrstVsk = 0.002
fMrstSk  = 0.012

# Distribution of resting cardiac output:
fCOrstCr  = 0.916
fCOrstMu  = 0.0492
fCOrstFat = 0.012
fCOrstVsk = 0.0228

# Tissue densities & conductivities:
Dfat  = 0.9
Dnfat = 1.1
kcr   = 54.29
kmu   = 41.76
kfat  = 15.87
kvsk  = 41.76
ksk   = 20.88
