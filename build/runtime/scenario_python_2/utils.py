# =============================================================================
# utils.py
#
# This Python module is a direct translation of the Java class "Utils"
# from the provided code. All functionality is intended to mirror the original
# Java version as closely as possible, line-by-line, preserving method names,
# logic, and comments.
#
# IMPORTANT:
#   - This code depends on certain classes and variables (FormatWriter, etc.)
#     that must be defined elsewhere or stubbed out in your Python environment
#     for it to be fully functional, especially for the file I/O routines.
#   - Where Java used StringBuffer, we have adapted these to Python strings.
#     Methods that filled a StringBuffer now return multiple values in a tuple
#     or build a Python string. We keep the same approach of returning integer
#     bit flags from parseURL to match the original design.
#   - The usage example at the bottom (showing how to import this file into
#     scenario_model.py) is abridged only. Everything else is full and
#     unabridged.
# =============================================================================

import math
import os
import datetime

# -----------------------------------------------------------------------------
# Constants (mirroring the Java private static final fields)
# -----------------------------------------------------------------------------
_idtbom = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

HRS_PER_CENTURY = 876600
HRS_PER_LEAP_YR = 8784
HRS_PER_NORM_YR = 8760
AVG_HRS_PER_YR  = 8766
CENTURY_PIVOT   = 50

YEAR  = 1
MONTH = 2
DAY   = 3
HOUR  = 4
SEC   = 5

NOT_A_URL = 1
NO_SERVER = 2
NO_FILE   = 4
NO_PORT   = 8
NO_PATH   = 16
NO_PAGE   = 32
SERVER_URL= 64
FILE_URL  = 128

LAT = 0
LON = 1


# -----------------------------------------------------------------------------
# parseURL
# -----------------------------------------------------------------------------
def parseURL(URL, server_out, path_out, page_out):
    """
    Parses a URL.
    Java method signature was:
      public static int parseURL(String URL, StringBuffer server1,
                                 StringBuffer path1, StringBuffer page1)

    Since Python doesn't use StringBuffer, we take the approach of:
    - 'server_out', 'path_out', 'page_out' are mutable lists or something that
      the caller provides, or we return an int plus a tuple of strings.

    For backward compatibility with the Java approach, we do:
      error_code = parseURL(URL, server_out, path_out, page_out)
    where server_out, path_out, page_out will be lists of length 1 to hold
    the string results.

    Return an integer error code (the bit flags).
    """
    retValue = 0

    # Clear the "string buffers" (here, we store them in index 0).
    server_out[0] = ""
    path_out[0] = ""
    page_out[0] = ""

    # Check for backslashes -> not a URL
    if '\\' in URL:
        retValue |= NOT_A_URL
        return retValue

    lenURL = len(URL)

    # URL type
    jfirst = URL.find('/')
    jnext = jfirst
    jlast = -1
    while jnext != -1:
        jlast = jnext
        jnext = URL.find('/', jlast + 1)
        if (jnext - jlast) != 1:
            break

    if (jlast - jfirst) < 1:
        retValue |= NOT_A_URL
        return retValue

    urlType = URL[:jlast + 1]
    urlType = urlType.upper()

    # Check if it's a FILE url
    if urlType.startswith("FILE"):
        # It's a file URL
        retValue |= FILE_URL

        jfirst = jlast + 1
        if jfirst == lenURL:
            retValue |= NO_FILE
            retValue |= NOT_A_URL
            return retValue

        jnext = jfirst
        jlast = -1
        while jnext != -1:
            jlast = jnext
            jnext = URL.find('/', jlast + 1)

        if jlast > jfirst:
            path = URL[jfirst:jlast + 1]
            path_out[0] = path
            jfirst = jlast + 1  # start of page
        else:
            retValue |= NO_FILE
            retValue |= NOT_A_URL
            return retValue

    else:
        # It's a server URL
        retValue |= SERVER_URL

        jfirst = jlast + 1
        if jfirst == lenURL:
            retValue |= NO_SERVER
            retValue |= NOT_A_URL
            return retValue

        jnext = jfirst
        jlast = jnext
        jnext = URL.find('/', jlast + 1)
        if jnext != -1:
            jlast = jnext

        if jlast > jfirst:
            server = URL[jfirst:jlast]
            # Check for port
            iport = server.find(':')
            if iport != -1:
                s = server[iport + 1:]
                port = int(s)  # may raise ValueError
                server = server[:iport]
                server_out[0] = server
            else:
                # If there's no port substring, we just put the entire server name
                server_out[0] = server
            jfirst = jlast
        else:
            retValue |= NO_SERVER
            retValue |= NOT_A_URL
            return retValue

        # path
        if jfirst == lenURL:
            retValue |= NO_PAGE
            retValue |= NOT_A_URL
            return retValue

        jnext = jfirst
        jlast = -1
        while jnext != -1:
            jlast = jnext
            jnext = URL.find('/', jlast + 1)

        if jlast > jfirst:
            path = URL[jfirst:jlast + 1]
            path_out[0] = path
            jfirst = jlast + 1
        else:
            # Possibly no path
            pass

    if jfirst == lenURL:
        retValue |= NO_PAGE
        retValue |= NOT_A_URL
        return retValue

    page = URL[jfirst:]
    page_out[0] = page

    return retValue


# -----------------------------------------------------------------------------
# urlToPath
# -----------------------------------------------------------------------------
def urlToPath(source):
    """
    Converts a URL string to a path by replacing '/' with the local file separator.
    Mirrors: public static String urlToPath(String source)
    """
    fs = os.sep  # System file separator
    s = source.replace('/', fs)
    return s


# -----------------------------------------------------------------------------
# DegsToDMS
# -----------------------------------------------------------------------------
def DegsToDMS(degs):
    """
    Converts decimal degrees to a degrees-minute-seconds string.
    Mirrors: public static String DegsToDMS(double degs)
    """
    negative = False
    if degs < 0:
        negative = True
        degs = -degs

    idegs = int(degs)
    ifrac = int((degs - float(idegs)) * 3600 + 0.5)
    mns = ifrac // 60
    secs = ifrac % 60

    # Build a char array
    result_chars = []

    if idegs >= 100:
        if negative:
            # total length = 11 in Java
            result_chars.append('-')
        else:
            # length = 10
            pass
        # '1' plus leftover degrees
        result_chars.append('1')
        idegs -= 100
    else:
        if negative:
            result_chars.append('-')

    # Next two digits for degrees
    digit = idegs // 10
    result_chars.append(chr(48 + digit))
    digit = idegs % 10
    result_chars.append(chr(48 + digit))
    result_chars.append('.')

    # minutes
    digit = mns // 10
    result_chars.append(chr(48 + digit))
    digit = mns % 10
    result_chars.append(chr(48 + digit))
    result_chars.append("'")

    # seconds
    digit = secs // 10
    result_chars.append(chr(48 + digit))
    digit = secs % 10
    result_chars.append(chr(48 + digit))
    result_chars.append('"')

    return "".join(result_chars)


# -----------------------------------------------------------------------------
# DMSToDegs
# -----------------------------------------------------------------------------
def DMSToDegs(dms, datum):
    """
    Converts a degrees-minute-seconds string to decimal degrees.
    A DMS string has the format: (-)ddd.mm'ss"
    @param dms a string to convert.
    @param datum the datum (LAT=0 or LON=1).
    @throws NumberFormatException in Java (here, we'll raise ValueError).
    """
    try:
        deg = dms.index(".")
        mns = dms.index("'")
        sec = dms.index('"')
    except ValueError as e:
        # If not found, we replicate the "Missing" checks in Java
        msg = "\nMissing degrees-minutes-separator or minutes-seconds separator"
        raise ValueError(msg)

    if deg == -1:
        raise ValueError("\nMissing degrees-minutes separator - ( . )")
    if mns == -1:
        raise ValueError("\nMissing minutes-seconds separator - ( ' )")
    if sec == -1:
        raise ValueError("\nMissing seconds terminator - ( \" )")

    s1 = dms[:deg]     # degrees
    s2 = dms[deg+1:mns]  # minutes
    s3 = dms[mns+1:sec]  # seconds

    negative = False
    val_deg = int(s1)
    if val_deg < 0:
        negative = True
        val_deg = -val_deg

    maxDatum = 90
    if datum == LON:
        maxDatum = 180

    if val_deg > maxDatum:
        raise ValueError("Degrees field OUT OF RANGE.")

    val_mns = int(s2)
    if val_mns < 0 or val_mns > 59:
        raise ValueError("Minutes field OUT OF RANGE.")

    val_sec = int(s3)
    if val_sec < 0 or val_sec > 59:
        raise ValueError("Seconds field OUT OF RANGE.")

    result = float(val_deg) + float(val_mns)/60.0 + float(val_sec)/3600.0
    if negative:
        result = -result

    return result


# -----------------------------------------------------------------------------
# isAllBlanks
# -----------------------------------------------------------------------------
def isAllBlanks(chArray):
    """
    Checks if the specified array of characters has all spaces in it.
    Mirrors: public static boolean isAllBlanks(char[] chArray)
    """
    for c in chArray:
        if c != ' ':
            return False
    return True


# -----------------------------------------------------------------------------
# HrsSince1900
# -----------------------------------------------------------------------------
def HrsSince1900(yr, mt, dy, hr, sec):
    """
    Computes hours since 1/1/1900.
    Java doc:
      yr - years since 1/1/1900
      mt - 1..12
      dy - 1..x
      hr - hhmm e.g. 1130
      sec - 0..59
    Returns double hours since 1900.
    Mirrors: public static double HrsSince1900(int yr, int mt, int dy, int hr, int sec)
    """
    ida = yr * 365 + ((yr + 3) // 4)
    idb = _idtbom[mt - 1] + dy - 1
    if (mt > 2) and ((yr % 4) == 0):
        idb += 1

    dHr = float((ida + idb) * 24)
    nHr = hr // 100
    nMn = hr % 100
    dMn = float(nMn) / 60.0
    dHr1 = float(nHr) + dMn + float(sec) / 3600.0
    return dHr + dHr1


# -----------------------------------------------------------------------------
# readProfileString
# -----------------------------------------------------------------------------
def readProfileString(fileObj, section, key, err_msg):
    """
    Reads a key value from a section in a file, using simplistic logic
    that parallels the Java version:
      public static String readProfileString(File file, String section,
                                             String key, String err_msg)

    We assume `fileObj` is a path to the file (string or Path).
    Returns the key value if successful, else err_msg.
    """
    file_path = str(fileObj)
    if not os.path.isfile(file_path):
        return err_msg

    s = err_msg  # assume failure
    sectionFound = False

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue
                if line.startswith('[') and sectionFound:
                    # we are now in a new section, stop
                    break
                if sectionFound:
                    if line.startswith(key):
                        index_eq = line.find('=')
                        if index_eq == -1:
                            break
                        key1 = line[:index_eq]
                        if key1 == key:
                            s = line[index_eq + 1:]
                            break
                else:
                    # look for section
                    if line.startswith(section):
                        sectionFound = True
    except Exception:
        pass

    return s


# -----------------------------------------------------------------------------
# writeProfileString
# -----------------------------------------------------------------------------
def writeProfileString(fileObj, section, key, value, err_msg):
    """
    Writes a key=value in a section to the file. Overwrites existing or
    appends if not found.

    Mirrors: public static String writeProfileString(File file, String section,
                        String key, String value, String err_msg)
    """
    file_path = str(fileObj)
    s = "key written"
    if not os.path.isfile(file_path):
        return err_msg

    dir_path = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    prefix, _sep, _suffix = basename.partition('.')
    try:
        import tempfile
        tmpFile = tempfile.NamedTemporaryFile(prefix=prefix, dir=dir_path, delete=False, mode='w', newline='')
        tmpFile_name = tmpFile.name
    except Exception:
        return err_msg

    sectionFound = False
    valueWritten = False

    try:
        with open(file_path, "r") as fin, tmpFile:
            lines = fin.readlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx].rstrip('\n')

                if not valueWritten:
                    if not sectionFound and line.startswith(section):
                        sectionFound = True
                        tmpFile.write(line + "\n")
                        idx += 1
                        continue

                    if sectionFound and line.startswith(key):
                        # write out new value
                        tmpFile.write(f"{key}={value}\n")
                        valueWritten = True
                        idx += 1
                        continue

                    # check if blank line inside section
                    if sectionFound and isAllBlanks(line):
                        tmpFile.write(f"{key}={value}\n")
                        tmpFile.write(line + "\n")
                        valueWritten = True
                        idx += 1
                        continue

                    # check if we've reached a new section
                    if sectionFound and line.startswith('['):
                        tmpFile.write(f"{key}={value}\n")
                        tmpFile.write(line + "\n")
                        valueWritten = True
                        idx += 1
                        continue

                tmpFile.write(line + "\n")
                idx += 1

            # end of file
            if sectionFound:
                if not valueWritten:
                    tmpFile.write(f"{key}={value}\n")
            else:
                s = err_msg

        # done writing
    except Exception:
        s = err_msg

    # rename
    try:
        os.remove(file_path)
        os.rename(tmpFile_name, file_path)
    except Exception:
        pass

    return s


# -----------------------------------------------------------------------------
# removeProfileString
# -----------------------------------------------------------------------------
def removeProfileString(fileObj, section, key, err_msg):
    """
    Removes a key=value pair in a section from the file.
    Mirrors: public static String removeProfileString(File file, String section,
                      String key, String err_msg)
    """
    file_path = str(fileObj)
    s = err_msg
    if not os.path.isfile(file_path):
        return err_msg

    dir_path = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    prefix, _sep, _suffix = basename.partition('.')
    try:
        import tempfile
        tmpFile = tempfile.NamedTemporaryFile(prefix=prefix, dir=dir_path, delete=False, mode='w', newline='')
        tmpFile_name = tmpFile.name
    except Exception:
        return err_msg

    sectionFound = False
    keyRemoved = False

    try:
        with open(file_path, "r") as fin, tmpFile:
            for line in fin:
                lineR = line.rstrip('\n')
                if not keyRemoved:
                    if not sectionFound and lineR.startswith(section):
                        sectionFound = True
                        tmpFile.write(lineR + "\n")
                        continue

                    if sectionFound and lineR.startswith(key):
                        # skip writing
                        keyRemoved = True
                        s = "key removed"
                        continue

                tmpFile.write(lineR + "\n")

    except Exception:
        pass

    # rename
    try:
        fin.close()
    except:
        pass

    try:
        tmpFile.close()
        os.remove(file_path)
        os.rename(tmpFile_name, file_path)
    except:
        pass

    return s


# -----------------------------------------------------------------------------
# timeIs
# -----------------------------------------------------------------------------
def timeIs():
    """
    Returns the current date-time in US locale style (approx).
    DateFormat.FULL, DateFormat.LONG in Java. Here we approximate.
    Mirrors: public static String timeIs()
    """
    # We'll produce a format somewhat similar to "Thursday, December 7, 2000 4:52:22 EST"
    # But Python doesn't have a direct "FULL" style. We'll do something approximate.
    now = datetime.datetime.now()
    # Example: "Thursday, December 07, 2000 16:52:22"
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


# -----------------------------------------------------------------------------
# scenarioTimeIs
# -----------------------------------------------------------------------------
def scenarioTimeIs():
    """
    Returns current date-time in US locale style (approx).
    Date has SHORT style, time has MEDIUM style in Java. We'll approximate.

    Mirrors: public static String scenarioTimeIs()
    """
    now = datetime.datetime.now()
    # "MM/DD/YY, HH:MM:SS"
    date_part = now.strftime("%m/%d/%y")
    time_part = now.strftime("%H:%M:%S")
    return f"{date_part}, {time_part}"


# -----------------------------------------------------------------------------
# DuBois
# -----------------------------------------------------------------------------
def DuBois(BW, HT):
    """
    DuBois formula: (.202 * BW^0.425 * (HT / 100)^0.725)
    Mirrors: public static float DuBois(float BW, float HT)
    """
    return 0.202 * (BW ** 0.425) * ((HT / 100.0) ** 0.725)


# -----------------------------------------------------------------------------
# SatVP
# -----------------------------------------------------------------------------
def SatVP(Temp):
    """
    Returns saturation vapor pressure in Torr at temperature Temp (C).
    Mirrors: public static float SatVP(float Temp)
    """
    Tk = 273.0 + float(Temp)
    LPsat = 28.59051 - (8.2 * math.log10(Tk)) + (0.00248 * Tk) - (3142.3 / Tk)
    Psat = 1000.0 * (10.0 ** LPsat)  # in mbars
    Psat *= 0.75  # in Torr
    return float(Psat)


# -----------------------------------------------------------------------------
# LagIt
# -----------------------------------------------------------------------------
def LagIt(OldVal, NewVal, time, HalfTime):
    """
    Calculates the Lag value.
    Mirrors: public static float LagIt(float OldVal, float NewVal, float time, float HalfTime)
    """
    return OldVal + (NewVal - OldVal) * (1.0 - math.exp(-0.693 * time / HalfTime))


# =============================================================================
# USAGE EXAMPLE (ABRIDGED) - how to import these functions in scenario_model.py:
#
#   from scenario_python_2.utils import (
#       parseURL, urlToPath, DegsToDMS, DMSToDegs, isAllBlanks, HrsSince1900,
#       readProfileString, writeProfileString, removeProfileString, timeIs,
#       scenarioTimeIs, DuBois, SatVP, LagIt
#   )
#
# Then you can call them, for example:
#   error_code = parseURL("http://example.com/home/index.html", server, path, page)
#   result = SatVP(37.0)
#
# =============================================================================
