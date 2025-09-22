import math
# This module contains the ellipsoid defining parameters

# WGS84 Ellipsoid Defining Parameters
a_84 = 6378137.000
inv_flatening = 1/298.257223536
e_84 = math.sqrt(2*inv_flatening - inv_flatening**2)
b_84 = a_84*(1- inv_flatening)
lnr_ecc_84 = math.sqrt((a_84**2 - b_84**2))
scn_ecc_84 = lnr_ecc_84/b_84

# War Office Ellipsoid Defining Parameters for local Datum
a_26 = 6378300
inv_flatening = 1/296
e_26 = math.sqrt(2*inv_flatening - inv_flatening**2)
b_26 = a_26*(1- inv_flatening)
lnr_ecc_26 = math.sqrt((a_26**2 - b_26**2))
scn_ecc_26 = lnr_ecc_26/b_26