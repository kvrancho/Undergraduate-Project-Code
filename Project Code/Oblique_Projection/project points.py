import math
import numpy as np
import pandas as pd
from library import *
from ellipsoids import *

# ---PROJECT WGS84 ELLIPSOIDAL COORDINATES TO XY COORDINATES USING OBLIQUE STEREOGRAPHIC PROJECTION SYSTEM---
# Read data into pandas dataframe
data = pd.read_csv('Controls Data.csv', header=None)
ctl = int(data.iat[0, 0]) # Get number of control stations

# Extract Local coordinates from data
local_xy = data.iloc[1:ctl+1, 0:4].copy()
local_xy.columns = ["Point ID", "Easting(m)", "Northing(m)", "Height(m)"]

# Extract GPS coordinates
gpsmat = data.iloc[(ctl + 1) : (2 * ctl + 1), 0 : data.shape[1]].copy()
gpsmat.columns = ['Point ID', 'LatD', 'LatMin', 'LatSec', 'LonD', 'LonMin', 'LonSec', 'Height']

# Create gps dataframe
header = ['Point ID', 'Latitude', 'Longitude', 'Height']
gps = pd.DataFrame(index=range(len(gpsmat)), columns=header)
for i in range(len(gps)):
    gps.loc[i,'Point ID'] = gpsmat.iat[i,0]
    gps.loc[i,'Latitude'] = radian(gpsmat.iat[i,1], gpsmat.iat[i,2], gpsmat.iat[i,3])
    gps.loc[i,'Longitude'] = radian(gpsmat.iat[i,4], gpsmat.iat[i,5], gpsmat.iat[i,6])
    gps.loc[i, 'Height'] = gpsmat.iat[i, 7]

# Extract Points to transform
ptsmat = data.iloc[(2 * ctl) + 1 : data.shape[0], 0 : data.shape[1]].copy()
ptsmat.columns = ['Point ID', 'LatD', 'LatMin', 'LatSec', 'LonD', 'LonMin', 'LonSec', 'Height']

# Create points to be transformed dataframe
header = ['Point ID', 'Latitude', 'Longitude', 'Height']
pts = pd.DataFrame(index=range(len(ptsmat)), columns=header)
for i in range(len(pts)):
    pts.loc[i, 'Point ID'] = ptsmat.iat[i, 0]
    pts.loc[i, 'Latitude'] = radian(ptsmat.iat[i, 1], ptsmat.iat[i, 2], ptsmat.iat[i, 3])
    pts.loc[i, 'Longitude'] = radian(ptsmat.iat[i, 4], ptsmat.iat[i, 5], ptsmat.iat[i, 6])
    pts.loc[i, 'Height'] = ptsmat.iat[i, 7]

# ------------------Convert GPS coordinates to oblique stereographic projection coordinates-----------
# Define origin of projection
lat_origin = sum(gps['Latitude'].to_list())/len(gps['Latitude'])
lon_origin = sum(gps['Longitude'].to_list())/len(gps['Longitude'])
hgt_origin = sum(gps['Height'].to_list())/len(gps['Height'])

# Compute Gaussian mean radius
R = math.sqrt(RM(a_84,e_84,lat_origin) * RN(a_84,e_84,lat_origin))

# Scale factor at Origin
k0 = 1 + (hgt_origin/R)

# Projection common functions
def L(e, lat):
    return 2 * math.atan(math.tan((math.pi/4) + (lat/2)) * ((1 - e*math.sin(lat))/(1 + e*math.sin(lat)))**(e/2)) - (math.pi/2)


def M(e, lat):
    return (math.cos(lat))/math.sqrt(1 - (e**2) * math.sin(lat)**2)


def A(a, k0, m0, chi_0, lon_0, chi, lon):
    return (2*a*k0*m0)/(math.cos(chi_0) * ((1 + math.sin(chi_0) * math.sin(chi) + math.cos(chi_0) * math.cos(chi) * math.cos(lon - lon_0))))


def E(A, lon_0, chi, lon):
    return A * (math.cos(chi) * math.sin(lon - lon_0))


def N(A, chi_0, lon_0, chi, lon):
    return A * ((math.cos(chi_0) * math.sin(chi) - math.sin(chi_0) * math.cos(chi) * math.cos(lon - lon_0)))


def K(A, chi, a, m):
    return (A * math.cos(chi))/(a * m)


# Function to convert GPS coordinates to 2D map coordinates
def oblique_stereo(a, e, D, P):
    # initialize NE matrix
    NE = np.zeros((D.shape[0], D.shape[1]), dtype=object)
    NE[:, 0] = D.iloc[:, 0]
    # compute mean values
    lat_0 = D.iloc[:, 1].mean()
    lon_0 = D.iloc[:, 2].mean()
    h0 = D.iloc[:, 3].mean()
    # scale factor
    k0 = 1 + h0/R     # Using Gaussian mean radius
    chi_0 = L(e,lat_0)
    m0 = M(e, lat_0)
    # Loop over rows of D
    for i in range(D.shape[0]):
        chi = L(e, D.iat[i, 1])
        m = M(e, D.iat[i, 1])
        aa = A(a, k0, m0, chi_0, lon_0, chi, D.iat[i, 2])
        NE[i, 1] = E(aa, lon_0, chi, D.iat[i, 2])
        NE[i, 2] = N(aa, chi_0, lon_0, chi, D.iat[i,2])
    NE[:, 3] = D.iloc[:, 3]  # copy heights

    # initialize XY matrix
    XY = np.zeros((P.shape[0], P.shape[1]), dtype=object)
    XY[:, 0] = P.iloc[:, 0]
    XY[:, 3] = P.iloc[:, 3]
    # Loop over rows of P
    for i in range(P.shape[0]):
        chi = L(e, P.iat[i, 1])
        m = M(e, P.iat[i, 1])
        aa = A(a, k0, m0, chi_0,lon_0, chi, P.iat[i,2])
        XY[i, 1] = E(aa, lon_0, chi, P.iat[i, 2])
        XY[i, 2] = N(aa, chi_0, lon_0, chi, P.iat[i, 2])

        # Convert to a proper DataFrame
        ctrlNE = pd.DataFrame(NE, columns=["Point ID", "Easting(m)", "Northing(m)", "Height(m)"])
        ptsXY = pd.DataFrame(XY, columns=["Point ID", "Easting(m)", "Northing(m)", "Height(m)"])
        ctrlNE[["Easting(m)", "Northing(m)", "Height(m)"]] = ctrlNE[["Easting(m)", "Northing(m)", "Height(m)"]].astype(float).round(5)
        ptsXY[["Easting(m)", "Northing(m)", "Height(m)"]] = ptsXY[["Easting(m)", "Northing(m)", "Height(m)"]].astype(float).round(5)
    return [ctrlNE, ptsXY]

# Get oblique stereographic projected coordinates for controls using their wgs84 coordinates
controls = oblique_stereo(a_84, e_84, gps, pts)[0]

# Get oblique stereographic projected coordinates for check points using their wgs84 coordinates
points_to_transformed = oblique_stereo(a_84, e_84, gps, pts)[1]

# Combine the local, controls, and points to transformed dataframes into one dataframe and drop Height column
combined_points = pd.concat([local_xy, controls, points_to_transformed], ignore_index=True).iloc[:, :-1]
combined_points["xstd.dev"] = 0
combined_points["ystd.dev"] = 0

df_headless = combined_points.to_numpy() # remove headers
new_row = [0] * df_headless.shape[1]   # same number of columns as df
df_headless = np.vstack([new_row, df_headless])
df_headless[0, 0] = ctl
df_headless[0, 1] = len(points_to_transformed)
two_dim_sta = pd.DataFrame(df_headless) # Convert back to dataframe

# Convert two dim dataframe into a csv file suitable for 2D Conformal Coordinates Transforms
two_dim_sta.to_csv("2D data file.csv", header=False, index=False)  # index=False prevents writing row numbers
