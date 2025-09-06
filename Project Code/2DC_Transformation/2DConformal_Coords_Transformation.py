import math
import pytz
import pandas as pd
import numpy as np
from Library import angle, azimuth
from datetime import datetime
from fpdf import FPDF
from Lines import *

# Get the current time in UTC
now_utc = datetime.now(pytz.utc)
ghana = str(pytz.timezone('GMT'))

# Register font type
FONT_CONSOLAS = "C:/Windows/Fonts/consola.ttf"

# Subclass FPDF to override the footer method
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Consolas", "", 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def output_df_to_pdf(pdf, df, title=""):
    col_widths = []
    for col in df.columns:
        max_content = max([len(str(val)) if pd.notna(val) else 0 for val in df[col]] + [len(col)])
        width = pdf.get_string_width('W' * max_content) + 4
        col_widths.append(width)

    table_width = sum(col_widths)
    page_width = pdf.w - 2 * pdf.l_margin
    x_start = pdf.l_margin + (page_width - table_width) / 2

    # --- Title ---
    if title:
        pdf.set_font('Consolas', 'B', 12)
        pdf.set_x(x_start)
        pdf.cell(table_width, 8, title, ln=1, align='C')
        line_y = pdf.get_y() - 1
        pdf.set_draw_color(0, 0, 0)
        pdf.line(x_start, line_y, x_start + table_width, line_y)
        pdf.line(x_start, line_y + 0.5, x_start + table_width, line_y + 0.5)
        pdf.ln(0)

    # --- Header ---
    pdf.set_font('Consolas', 'B', 12)
    pdf.set_x(x_start)
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 6, col, border='B', align='C')
    pdf.ln(6)

    # --- Data rows ---
    pdf.set_font('Consolas', '', 12)
    for row_index, row in enumerate(df.itertuples(index=False)):
        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()
            # Repeat header on new page
            pdf.set_font('Consolas', 'B', 12)
            pdf.set_x(x_start)
            for i, col in enumerate(df.columns):
                pdf.cell(col_widths[i], 6, col, border='B', align='C')
            pdf.ln(6)
            pdf.set_font('Consolas', '', 12)

        pdf.set_x(x_start)
        for i, value in enumerate(row):
            cell_text = "" if pd.isna(value) else str(value)
            border_style = 'B' if row_index == len(df) - 1 else 0
            pdf.cell(col_widths[i], 6, cell_text, border=border_style, align='C')
        pdf.ln(6)

def load_font(pdf):
    try:
        pdf.add_font("Consolas", "", FONT_CONSOLAS, uni=True)
        pdf.add_font("Consolas", "B", FONT_CONSOLAS, uni=True)
        pdf.set_font("Consolas", "", 12)
    except Exception as e:
        print("Warning: Consolas font not found. Using Courier.")
        print("Details:", e)
        pdf.set_font("Courier", "", 12)
    return pdf

# --- Main Execution ---
pdf = PDF()  # use the subclass with footer
pdf.set_auto_page_break(auto=True, margin=15)
pdf.alias_nb_pages()

load_font(pdf)

pdf.add_page()

pdf.cell(0, 5, "            GeoEye Geospatial Consult       Time in UTC: " + now_utc.strftime("%Y-%m-%d %H:%M:%S"), ln=True)
pdf.cell(0, 5, "            +233249219236/+233208982112       Time Zone: " + ghana, ln=True)
pdf.cell(0, 5, "            geoeyegeospatial@gmail.com      Country: Ghana", ln=True)
pdf.ln(4)
pdf.cell(0,5, '                        2D CONFORMAL COORDINATES TRANSFORMATION', ln=True)
add_double_line(pdf, 0, 0.5, 25, 25)
pdf.ln(1)
pdf.cell(0, 5, '                        Model Equations: ax - by + Tx = X + Vx', ln=True)
pdf.cell(0, 5, '                                         bx + ay + Ty = Y + Vy', ln=True)
pdf.ln(1)
add_double_line(pdf, 0, 0.5, 25, 25)
pdf.ln(2)

# Read data into a pandas dataframe
data = pd.read_csv('Data.csv',header=None)
pairs = int(data.iat[0,0]) # Number of controls/common points
transpts = int(data.iat[0,1]) # Number of points to be transformed
pdf.cell(0, 5, f'                     Number of common points or controls: {pairs}', ln=True)
pdf.cell(0, 5, f'                     Number of points to be transformed:  {transpts}', ln=True)
pdf.ln(1)
add_double_line(pdf, 0, 0.5, 25, 25)
pdf.ln(3)

# Create new dataframe for points and fill it
header = ['Point', 'x', 'y', 'Sx', 'Sy','Easting', 'Northing']
Pts = pd.DataFrame(index=range(pairs), columns=header).astype(object)
for i in range(1, pairs + 1):
    for j in range(1, 3):
        Pts.iat[i - 1, 0] = data.iat[i, 0] # Ids
        Pts.iat[i - 1, j+4] = data.iat[i,j] # Controls coordinates
        Pts.iat[i - 1, j] = data.iat[i + pairs, j] # Measured coordinates

#The following statement set the standard deviations for the measured coordinates
#to the user entry value for a weighted adjusted or 1 for an unweighted adjustment.
for i in range(1, pairs + 1):
    if data.iat[i + pairs, 3] != 0:
        Pts.iat[i - 1, 3] = data.iat[i + pairs, 3]
    else:
        Pts.iat[i - 1, 3] = 1

    if data.iat[i + pairs, 4] != 0:
        Pts.iat[i - 1, 4] = data.iat[i + pairs, 4]
    else:
        Pts.iat[i - 1, 4] = 1

# Add measured and control data to pdf
output_df_to_pdf(pdf, df=Pts, title='MEASURED(xy) AND CONTROL DATA')
pdf.ln(3)

# Get measured points to be transformed
mdata = data[2 * pairs + 1:]
header = ['Point', 'x', 'y', 'Sx', 'Sy']
mdata.columns = header
new_mdata = pd.DataFrame(index=range(transpts), columns=header).astype(object)
new_mdata['Point'] = mdata['Point'].to_list()
new_mdata['x'] = mdata['x'].to_list()
new_mdata['y'] = mdata['y'].to_list()
new_mdata['Sx'] = mdata['Sx'].to_list()
new_mdata['Sy'] = mdata['Sy'].to_list()

# Add measured data to pdf
output_df_to_pdf(pdf, df=new_mdata, title='POINTS TO BE TRANSFORMED')
pdf.ln(1)
pdf.cell(0, 5, '             Sx and Sy are the uncertainties in the measured points', ln=True)
pdf.ln(3)

# Build Matrices
def BldMat(Pts, pairs):
    A = np.zeros((2 * pairs, 4)).astype(float)
    L = np.zeros((2 * pairs, 1)).astype(float)
    W = np.zeros((2 * pairs, 2 * pairs)).astype(float)
    for i in range(pairs):
        r1 = 2 * i
        r2 = 2 * i + 1
        A[r1, 0] = Pts.iat[i, 1]
        A[r1, 1] = -Pts.iat[i, 2]
        A[r1, 2] = 1
        A[r1, 3] = 0
        L[r1, 0] = Pts.iat[i, 5]

        A[r2, 0] = Pts.iat[i, 2]
        A[r2, 1] = Pts.iat[i, 1]
        A[r2, 2] = 0
        A[r2, 3] = 1
        L[r2, 0] = Pts.iat[i, 6]

        # Assign weights
        for j in range(2 * pairs):
            W[r1, j] = 0 # Initializes weight matrix to zeros
            W[r2, j] = 0
        W[r1, r1] = 1 / (Pts.iat[i, 3]) ** 2
        W[r2, r2] = 1 / (Pts.iat[i, 4]) ** 2

    return [A, L, W]

A = BldMat(Pts, pairs)[0]
L = BldMat(Pts, pairs)[1]
W = BldMat(Pts, pairs)[2]

# Least squares Solutions
N = np.transpose(A) @ W @ A
M = np.transpose(A) @ W @ L
Qxx = np.linalg.inv(N)
X = Qxx @ M

# Adjusted Observations ( i.e Calculated control coordinates using Transformation parameters)
Adj_obs = A @ X

# Compute Residuals, V
V = Adj_obs - L

# Compute variance
numerator = np.transpose(V) @ W @ V
degree_freedom = (2 * pairs) - 4
variance = (numerator[0])/degree_freedom
std_unit_weight = math.sqrt(variance[0])

# Propagate errors to adjusted observation
Qll = A @ Qxx @ np.transpose(A)

# Adjusted Parameters
a = X[0][0]
b = X[1][0]
Tx = X[2][0] #Translation in the x-direction
Ty = X[3][0] #Translation in the y-direction

# Their standard deviations or errors
Sa = std_unit_weight * math.sqrt(Qxx[0,0])
Sb = std_unit_weight * math.sqrt(Qxx[1,1])
STx = std_unit_weight * math.sqrt(Qxx[2,2])
STy = std_unit_weight * math.sqrt(Qxx[3,3])

# Their t-statistic
ta = abs(a)/Sa
tb = abs(b)/Sb
tx = abs(Tx)/STx
ty = abs(Ty)/STy

# Compute Orientation(rotation) angle and Scale factor
theta = math.atan(b/a)
scale = a/math.cos(theta)
theta_dms = angle(azimuth(b, a),1)

# Create dataframe for transformation parameters
header = ['Parameter', 'Value', 'Standard Deviation(±)', 't-value']
trans_param = pd.DataFrame(index=range(4), columns=header).astype(object)
# Fill Parameter column
trans_param.loc[0, 'Parameter'] = 'a'
trans_param.loc[1, 'Parameter'] = 'b'
trans_param.loc[2, 'Parameter'] = 'Tx'
trans_param.loc[3, 'Parameter'] = 'Ty'
# Fill Value column
trans_param.loc[0, 'Value'] = f'{float(a):,.3f}'
trans_param.loc[1, 'Value'] = f'{float(b):,.3f}'
trans_param.loc[2, 'Value'] = f'{float(Tx):,.3f}'
trans_param.loc[3, 'Value'] = f'{float(Ty):,.3f}'
# Fill Standard Deviation column
trans_param.loc[0, 'Standard Deviation(±)'] = f'{float(Sa):,.3f}'
trans_param.loc[1, 'Standard Deviation(±)'] = f'{float(Sb):,.3f}'
trans_param.loc[2, 'Standard Deviation(±)'] = f'{float(STx):,.3f}'
trans_param.loc[3, 'Standard Deviation(±)'] = f'{float(STy):,.3f}'
# Fill t-value column
trans_param.loc[0, 't-value'] = f'{float(ta):,.3f}'
trans_param.loc[1, 't-value'] = f'{float(tb):,.3f}'
trans_param.loc[2, 't-value'] = f'{float(tx):,.3f}'
trans_param.loc[3, 't-value'] = f'{float(ty):,.3f}'

output_df_to_pdf(pdf, df=trans_param, title='TRANSFORMATION PARAMETERS AND THEIR T-STATISTIC')
pdf.ln(1)
pdf.cell(0, 5, f'                            Scale factor: {scale:>12.8f}', ln=True)
pdf.cell(0, 5, f'                          Rotation angle: {theta_dms:>12}', ln=True)
pdf.ln(1)
add_double_line(pdf, 0, 0.5, 35, 35)
pdf.ln(4)

# Display Transformation Statistics
pdf.cell(0, 5, '                            TRANSFORMATION STATISTICS', ln=True)
add_double_line(pdf, 0, 0.5, 35, 35)
pdf.ln(1)
pdf.cell(0, 5, f'                          Number of equations: {len(A)}', ln=True)
pdf.cell(0, 5, f'                           Number of unknowns: {len(A) - degree_freedom}', ln=True)
pdf.cell(0, 5, f'                            Degree of freedom: {degree_freedom}', ln=True)
pdf.cell(0, 5, f'                                     Variance: {variance[0]:.8f} sqr.unit', ln=True)
pdf.cell(0, 5, f'                           Standard deviation: ±{std_unit_weight:.8f} unit', ln=True)
pdf.ln(1)
add_double_line(pdf, 0, 0.5, 35, 35)
pdf.ln(3)

# Calculated Control Coordinates using Transformation results.
header = ['Point', 'Easting', 'Northing', 'Ve', 'Vn', 'σE', 'σN']
results = pd.DataFrame(index=range(pairs), columns=header).astype(object)
for i in range(pairs):
    results.loc[i, 'Point'] = Pts.iat[i, 0]
    results.loc[i, 'Easting'] = f'{float(Adj_obs[2 * i, 0]):,.3f}'
    results.loc[i, 'Northing'] = f'{float(Adj_obs[2 * i + 1, 0]):,.3f}'
    results.loc[i, 'Ve'] = f'{float(V[2 * i, 0]):,.3f}'
    results.loc[i, 'Vn'] = f'{float(V[2 * i + 1, 0]):,.3f}'
    results.loc[i, 'σE'] = f'{float(std_unit_weight * math.sqrt(Qll[2 * i, 2 * i])):,.3f}'
    results.loc[i, 'σN'] = f'{float(std_unit_weight * math.sqrt(Qll[(2 * i) + 1, (2 * i) + 1])):,.3f}'
output_df_to_pdf(pdf, df=results, title='CALCULATED CONTROL COORDINATES USING PARAMETERS')
pdf.ln(3)

# Transform Other Points. Thus loop through points to transform and build A matrix
def BldATrans(data, pairs, transpts):
    A = np.zeros((2 * transpts, 4)).astype(float)
    k = 2 * pairs
    r = 0
    for i in range(transpts):
        c = k + i + 1
        A[r, 0] = data.iat[c, 1]
        A[r, 1] = -data.iat[c, 2]
        A[r, 2] = 1
        A[r, 3] = 0
        r = r + 1
        c = k + i + 1
        A[r, 0] = data.iat[c, 2]
        A[r, 1] = data.iat[c, 1]
        A[r, 2] = 0
        A[r, 3] = 1
        r = r + 1
    return A
A = BldATrans(data, pairs, transpts)

# Transformed Points and Covariance matrix for adjusted observations
adj_obs = A @ X
Qll = variance * (A @ Qxx @ np.transpose(A))
header = ['Point', 'Easting', 'Northing', 'σE', 'σN']
TransResults = pd.DataFrame(np.zeros((transpts, len(header))), columns=header).astype(object)
for i in range(transpts):
    TransResults.iat[i, 0] = data.iat[2 * pairs + 1 + i, 0] # Ids of points
    TransResults.iat[i, 1] = f'{float(adj_obs[2 * i, 0]):,.3f}' # Easting coords
    TransResults.iat[i, 2] = f'{float(adj_obs[2 * i + 1, 0]):,.3f}' # Northing coords
    TransResults.iat[i, 3] = f'{float(math.sqrt(Qll[2 * i, 2 * i])):,.3f}' # Set Easting coords uncertainty
    TransResults.iat[i, 4] = f'{float(math.sqrt(Qll[2 * i + 1, 2 * i + 1])):,.3f}' # Set Northing coords uncertainty

output_df_to_pdf(pdf, df=TransResults, title='TRANSFORMED COORDINATES')
pdf.cell(1)
pdf.cell(0, 5, f'   --------------------------------Finished!--------------------------------', ln=True)

# Save Pdf file
pdf.output('2DC Output.pdf')
print('Done!')
print('2DC Output.pdf Successfully Created!')