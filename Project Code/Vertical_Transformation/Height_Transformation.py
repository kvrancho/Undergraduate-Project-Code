import math
import numpy as np
import pandas as pd
from library import *
from ellipsoids import *

# ----------------- TRANSFORMATION OF ELLIPSOIDAL HEIGHTS TO ORTHOMETRIC HEIGHT ---------------
data = pd.read_csv(".\HData.csv", header=None)
npts = int(data.iat[0, 0])
nsta = int(data.iat[0, 1])

# Extract Local coordinates from data
local = data.iloc[1:npts+1, 0:4].copy()
local.columns = ["Point ID", "Easting(m)", "Northing(m)", "Height(m)"]

# Extract GPS coordinates
gpsmat = data.iloc[(npts + 1) : (2 * npts + nsta + 1), 0 : data.shape[1]].copy().iloc[:, :-1]
gpsmat.columns = ['Point ID', 'LatD', 'LatMin', 'LatSec', 'LonD', 'LonMin', 'LonSec', 'Height']
gpsmat.index = range(len(gpsmat))

# Parse GPS data into latitude, longitude and height
header = ['Point ID', 'Latitude', 'Longitude', 'globalH']
pts = pd.DataFrame(index=range(len(gpsmat)), columns=header)
for i in range(len(gpsmat)):
    pts.loc[i, 'Point ID'] = gpsmat.iat[i, 0]
    pts.loc[i, 'Latitude'] = radian(gpsmat.iat[i, 1], gpsmat.iat[i, 2], gpsmat.iat[i, 3])
    pts.loc[i, 'Longitude'] = radian(gpsmat.iat[i, 4], gpsmat.iat[i, 5], gpsmat.iat[i, 6])
    pts.loc[i, 'globalH'] = gpsmat.iat[i, 7]

# Determine the coordinates of the centroid
lat_centroid = pts['Latitude'].mean()
lon_centroid = pts['Longitude'].mean()
hght_centroid = pts['globalH'].mean()

# Compute scale factor for oblique stereographic map projection
R = math.sqrt(RM(a_84,e_84,lat_centroid) * RN(a_84,e_84,lat_centroid)) # Gaussian mean radius
k0 = 1 + (hght_centroid/R) # Scale factor at origin

# Projection common functions
def L(e, lat):
    return 2 * math.atan(math.tan((math.pi/4) + (lat/2)) * ((1 - e*math.sin(lat))/(1 + e*math.sin(lat)))**(e/2)) - (math.pi/2)

def M(e, lat):
    return (math.cos(lat))/math.sqrt(1 - (e**2) * math.sin(lat)**2)

def E(A, lon_0, chi, lon):
    return A * (math.cos(chi) * math.sin(lon - lon_0))

def N(A, chi_0, lon_0, chi, lon):
    return A * ((math.cos(chi_0) * math.sin(chi) - math.sin(chi_0) * math.cos(chi) * math.cos(lon - lon_0)))

def K(A, chi, a, m):
    return (A * math.cos(chi))/(a * m)


def A(a, k0, m0, chi_0, lon_0, chi, lon):
    return (2*a*k0*m0)/(math.cos(chi_0) * ((1 + math.sin(chi_0) * math.sin(chi) + math.cos(chi_0) * math.cos(chi) * math.cos(lon - lon_0))))

chi_0 = L(e_84, lat_centroid)
m_0 = M(e_84, lat_centroid)

# Direct problem
header = ['Point ID', 'Easting', 'Northing', 'globalH']
proj_coords = pd.DataFrame(index=range(len(pts)), columns=header)
for i in range(len(proj_coords)):
    chi = L(e_84, pts.loc[i, 'Latitude'])
    m = M(e_84, pts.loc[i, 'Latitude'])
    l = pts.loc[i, 'Longitude']
    AA = A(a_84, k0, m_0, chi_0, lon_centroid, chi, l)
    proj_coords.loc[i, 'Point ID'] = pts.loc[i, 'Point ID']
    proj_coords.loc[i, 'Easting'] = E(AA, lon_centroid, chi, l)
    proj_coords.loc[i, 'Northing'] = N(AA, chi_0, lon_centroid, chi, l)
    proj_coords.loc[i, 'globalH'] = pts.loc[i, 'globalH']

# Compute centroid coordinates in local reference system
xcentroid = local['Easting(m)'].mean()
ycentroid = local['Northing(m)'].mean()

# Compute centroid coordinates for GPS easting and northing coordinates
sum_e = 0
sum_n = 0
for i in range(npts):
    sum_e += proj_coords.loc[i, 'Easting']
    sum_n += proj_coords.loc[i, 'Northing']
ecentroid = sum_e/npts
ncentroid = sum_n/npts

# Translate local coordinates to centroid
x = []
y = []
for i in range(1, npts+1):
    x.append(local.loc[i, 'Easting(m)'] - xcentroid)
    y.append(local.loc[i, 'Northing(m)'] - ycentroid)


# Translate GPS eastings and northings to the centroids position
e = []
n = []
for i in range(npts):
    e.append(proj_coords.loc[i, 'Easting'] - ecentroid)
    n.append(proj_coords.loc[i, 'Northing'] - ncentroid)

# Add GPS Points to transformed to e and n arrays
for i in range(npts, npts + nsta):
    e.append(proj_coords.loc[i, 'Easting'])
    n.append(proj_coords.loc[i, 'Northing'])


def BuiltMat(n, e, hl, hg, npts):
    A = np.zeros((npts, 3), dtype=float)
    L = np.zeros((npts, 1), dtype=float)
    for i in range(npts):
        A[i, 0] = 1
        A[i, 1] = n[i]
        A[i, 2] = e[i]
        L[i, 0] = hl[i] - hg[i]

    return [A, L]

# Convert Local Height into list.
lH = []
for i in range(1, npts+1):
    lH.append(local.loc[i, 'Height(m)'])

# Convert global height into list
gH = []
for i in range(npts):
    gH.append(proj_coords.loc[i, 'globalH'])

# Get matrices A and L
A = BuiltMat(n, e, lH, gH, npts)[0]
L = BuiltMat(n, e, lH, gH, npts)[1]

# Solve for vertical Transformation parameters
N = np.transpose(A) @ A
M = np.transpose(A) @ L
Qxx = np.linalg.inv(N)
X = Qxx @ M

T0 = X[0, 0]  # Shift or Translation
re = X[1, 0]  # Rotation about the east direction
rn = X[2, 0]  # Rotation about the north direction

# Compute Residuals
V = (A @ X) - L

# Extract points to transform
points = proj_coords.iloc[npts:npts+nsta, 0:4].reset_index(drop=True)

# Create dataframe for transformed points
header = ['Point ID', 'Ellipsoidal Height(h)', 'Geiodal Height(N)', 'Orthomertric Height(H)']
transformed_points = pd.DataFrame(index=range(nsta), columns=header)

# Transformed remaining GPS Ellipsoidal Heights to Orthometric Heights
for i in range(len(transformed_points)):
    geoidal_height = re*float(points.loc[i,'Northing']) + rn*float(points.loc[i,'Easting']) + T0
    transformed_points.loc[i, 'Point ID'] = points.loc[i, 'Point ID']
    transformed_points.loc[i, 'Ellipsoidal Height(h)'] = points.loc[i, 'globalH']
    transformed_points.loc[i, 'Geiodal Height(N)'] = round(geoidal_height,3)
    transformed_points.loc[i, 'Orthomertric Height(H)'] = float(points.loc[i, 'globalH']) + round(geoidal_height,3)

# Diplay transformation results
print(transformed_points)