#This Library contains useful functions which other program depends upon.
#All routines are kept herein for the purpose of program efficiency and readability
#You must understand how they work before using them. However, you may feel free to use them
import math

rho = (180*3600)/math.pi

def radian(d, m, s):
    """Returns the radian value of an angle or direction in degree, minutes and seconds."""
    if abs(d) > 0:
        i = math.copysign(1, d)
    elif abs(m) > 0:
        i = math.copysign(1, m)
    else:
        i = math.copysign(1, s)
    return i*(abs(d) + abs(m)/60 + abs(s)/60**2)*(math.pi/180)

def rad2dms(x, n):
    """Converts a radian angle into D M S format with n decimal places."""
    dd = abs((180/math.pi) * x)
    i = math.copysign(1, x)
    d = math.floor(dd)
    m = math.floor((dd - d) * 60)
    s = ((dd - d - m/60) * 3600)
    s = round(s, n)
    d = d * i
    m = m * i
    s = s * i
    return f"{int(d)}{'u00b0'} {int(m)}{'u2032'} {s}{'u2033'}"

def angle(x, n):
    """Takes an angle in radian units and returns the formated angle in ddd mm ss.ssss"""
    dd = abs(180/math.pi * x)
    sign = math.copysign(1, x)
    d = math.floor(dd)
    m = math.floor((dd - d) * 60)
    s = (dd - d - m/60)*3600
    o = 5*10**-(n + 1)
    if (s + o) >= 60:
        s = s - 60
        m = m + 1
    if m >= 60:
        m = m - 60
        d = d + 1
    s = round(s, n)

    # Handle the case where s is exactly 0 after rounding
    if s == 0:
        s_str = '00'
    else:
        s_str = f"{s:02.{n}f}" if n > 0 else f"{s:02.0f}"

    if dd < 10:
        dstr = '00' + str(d) + '\u00B0 '
    elif dd < 100:
        dstr = '0' + str(d) + '\u00B0 '
    else:
        dstr = str(d) + '\u00B0 '

    if m < 10:
        mstr = '0' + str(m) + '\u2032 '
    else:
        mstr = str(m) + '\u2032 '

    sstr = s_str + '\u2033'

    if sign < 0:
        s1 = f"{'-'}{dstr}{mstr}{sstr}"
    else:
        s1 = f"{dstr}{mstr}{sstr}"
    return s1

def geoangle(x, n, coord_type=None):
    """Takes an angle in radian units and returns the formatted angle in ddd mm ss.ssss
    with optional N/S/E/W direction based on coord_type ('lat' or 'lon'). If
    coord_type is not provided, angle will just return the formated angle"""
    dd = abs(180 / math.pi * x)
    sign = math.copysign(1, x)
    d = math.floor(dd)
    m = math.floor((dd - d) * 60)
    s = (dd - d - m / 60) * 3600
    o = 5 * 10**-(n + 1)

    if (s + o) >= 60:
        s = s - 60
        m = m + 1

    if m >= 60:
        m = m - 60
        d = d + 1

    s = round(s, n)

    # Handle the case where s is exactly 0 after rounding
    if s == 0:
        s_str = '00'
    else:
        s_str = f"{s:02.{n}f}" if n > 0 else f"{s:02.0f}"

    if dd < 10:
        dstr = '0' + str(d) + '\u00B0 '
    elif dd > 10:
        dstr = str(d) + '\u00B0 '
    else:
        dstr = str(d) + '\u00B0 '
    if m < 10:
        mstr = '0' + str(m) + '\u2032 '
    else:
        mstr = str(m) + '\u2032 '
    if s < 10:
        sstr = '0' + str(s)
    else:
        sstr = str(s)
    if n > 0:
        c = n + 3 - len(sstr)
        if len(sstr) == 2:
            sstr = str(sstr) + '.'
            c = c - 1
        if c != 0:
            for j in range(c):
                sstr = str(sstr) + '0'

    if sign < 0:
        s1 = dstr + mstr + sstr
    else:
        s1 = dstr + mstr + sstr

    # Append direction label if coord_type is specified
    if coord_type == 'lat':
        if sign > 0:
            s1 = s1 + '\u2033' + ' N'
        else:
            s1 = s1 + '\u2033' + ' S'
    elif coord_type == 'lon':
        if sign > 0:
            s1 = s1 + '\u2033' + ' E'
        else:
            s1 = s1 + '\u2033' + ' W'

    return s1

def RN(a, e, lat):
    """Compute radius of the curvature at a given latitude"""
    return a / (math.sqrt(1 - (e**2) * math.sin(lat)**2))


def RM(a, e, lat):
    """Compute radius of the meridian at a given latitude"""
    return a*(1 - e**2)/(1 - (e**2)*math.sin(lat)**2)**(3/2)


def X(a, e, lat, long, h):
    """Compute and returns the geocentric cartesian X coordinates"""
    return (RN(a, e, lat) + h) * math.cos(lat) * math.cos(long)


def Y(a, e, lat, long, h):
    """Compute and returns the geocentric cartesian Y coordinates"""
    return (RN(a, e, lat) + h) * math.cos(lat) * math.sin(long)


def Z(a, e, lat, long, h):
    """Compute and returns the geocentric cartesian Z coordinates"""
    return (RN(a, e, lat) * (1 - (e**2)) + h) * math.sin(lat)


def latitude(a, e, X, Y, Z):
    """Returns the geodetic latitude of point in radians."""
    D = math.sqrt(X**2 + Y**2)
    p_lat = math.pi
    lat = math.atan(Z/(D*(1 - e**2)))
    while (abs(p_lat - lat) > 0.000000000001):
        p_lat = lat
        N = RN(a, e, lat)
        lat = math.atan((Z + e**2 * N * math.sin(p_lat))/D)
        return lat


def longitude(X, Y):
    """Returns the geodetic longitude of a point in radians."""
    return math.atan2(Y, X)


def height(a, e, lat, X, Y, Z):
    """Returns the ellipsoidal height of a point in meters."""
    D = math.sqrt(X**2 + Y**2)
    if (abs(lat) < math.pi/4):
        return (D/math.cos(lat)) - RN(a, e, lat)
    else:
        (Z/math.sin(lat)) - RN(a, e, lat) * (1 - e**2)