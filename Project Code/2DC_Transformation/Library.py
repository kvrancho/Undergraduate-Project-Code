#This Library contains useful functions which other program depends upon.
#All routines are kept herein for program efficiency and readability
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

def distance(dx, dy):
    """Returns the distance of a course based on dx, dy"""
    return math.sqrt(dx**2 + dy**2)

def azimuth(dx, dy):
    """Returns the azimuth of a course base on dx, dy"""
    a = math.atan2(dx, dy)
    if a < 0:
        a = a + 2 * math.pi
    return a

def normalize(a):
    while a >= 2 * math.pi:
        a = a - 2 * math.pi
    while a < 0:
        a = a + 2 * math.pi
    return a

def compute_angle(a_b, a_f):
    if a_f > a_b:
        a = a_f - a_b
    else:
        a = 2 * math.pi + a_f - a_b
    return normalize(a)


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

    if s < 10:
        sstr = '0' + str(s) + '\u2033'
    else:
        sstr = str(s) + '\u2033'

    if sign < 0:
        s1 = f"{'-'}{dstr}{mstr}{sstr}"
    else:
        s1 = f"{dstr}{mstr}{sstr}"
    return s1

def area(x, y):
    x.append(x[0])
    y.append(y[0])
    n = len(x)
    plus = 0
    minus = 0
    for i in range(n - 1):
        plus += x[i] * y[i + 1]
        minus += y[i] * x[i + 1]
    return 0.5 * abs(plus - minus)

def brg_angle(x, n):
    """Takes an angle in radian units and returns the formated angle in dd mm ss.ssss"""
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
        dstr = '0' + str(d) + '\u00B0 '
    else:
        dstr = str(d) + '\u00B0 '

    if m < 10:
        mstr = '0' + str(m) + '\u2032 '
    else:
        mstr = str(m) + '\u2032 '

    if s < 10:
        sstr = '0' + str(s) + '\u2033'
    else:
        sstr = str(s) + '\u2033'

    if n > 0:
        c = n + 3 - len(sstr)
        if len(sstr) == 2:
            sstr = sstr + '.'
            c = c - 1
        if c != 0:
            for j in range(c):
                sstr = sstr + '0'

    if sign < 0:
        s1 = f"{'-'}{dstr}{mstr}{sstr}"
    else:
        s1 = f"{dstr}{mstr}{sstr}"
    return s1

def bearing(a, n):
    """Takes an azimuth in radians and return its bearing
        with n decimal places of the seconds value
    """

    a = normalize(a)
    d = a * (180/math.pi)

    if d > 90 and d <= 270:
        f = 'S'
    else:
        f = 'N'

    if d <= 180:
        l = 'E'
    else:
        l = 'W'

    if 90 < d <= 180:
        a = math.pi - a
    else:
        if 180 < d < 270:
            a = a - math.pi
        else:
            if d >= 270:
                a = 2*math.pi - a
    b = brg_angle(a,n)
    return f"{f} {b} {l}"

def degmin_angle(x, n):
    """Takes an angle in radian units and returns the formated angle in ddd mm"""
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

    if s >= 30:
        m += 1

    if m == 60:
        d += 1

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

    if sign < 0:
        s1 = f"{'-'}{dstr}{mstr}"
    else:
        s1 = f"{dstr}{mstr}"
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
        dstr = '00' + str(d) + '\u00B0 '
    elif dd < 100:
        dstr = '0' + str(d) + '\u00B0 '
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