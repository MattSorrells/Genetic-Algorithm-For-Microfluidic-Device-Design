import numpy as np
from numpy import linalg
import pandas as pd
# ----- Further abstracted versions of the HCA
def hct_error(lengths,hct):
    rev_design_dP=np.linspace(.5,5,4)
    design_dP=rev_design_dP[::-1]
    df=HCA_from_spacing(lengths)
    df=df[df['Channel'].str.contains("Inj")]
    #     print(df)
    dP_act=df['dP (kPa)'].values
    #     print(dP_act)
    error=np.zeros(len(dP_act))
    for i in range(len(error)):
        error[i]=np.absolute(dP_act[i]-design_dP[i])/design_dP[i]*100
    injury=[1,2,3,4]
    err_df=pd.DataFrame({'% Error':error,'Injury':injury})
    err_df['HCT']=[hct]*len(err_df)
    return err_df

def HCA_from_spacing(lengths):
    entL=4.9 # entrance length in bleeding chip
    blood_in=4.9
    injL=150e-3 # injury width
    inj_space1=lengths[0] # distance between injury channels
    inj_space2=lengths[1] # distance between injury channels
    inj_space3=lengths[2] # distance between injury channels
    BO_resL=lengths[3] # resistor on the blood outlet channel
    WO_resL=lengths[4] # resistor on wash side
    tubL=30 # length of tubing + flow meters
    fun_lengths=[blood_in,entL,injL,inj_space3,inj_space1,injL,inj_space2,inj_space2,injL,
                                         inj_space1,inj_space3,injL,BO_resL,WO_resL,tubL,tubL]
    df=HCA_BC_Multiple_Inj_Counter(lengths=fun_lengths)
    return df

# --------- Main function of the program
# -----------___ inputs -
# Channel key:
# Channel	    0	        1	      2	      3	      4	       5	   6	   7	  8
# Description	Blood In	Wash In	Inj. 1	Wash 2	Blood 2	Inj. 2 	Blood 3	Wash 3	Inj 3
#     9	      10	  11	  12	        13	1           4	           15
# 	Wash 4	Blood 4	Inj 4	Blood Out Res	Wash Out Res	Blood Out Tub. 	Wash Out Tub
# ------ Dimensions -------
# Lengths (mm):
entL = 4.9  # entrance length in bleeding chip
injL = 150e-3  # injury width
inj_space = 10  # distance between injury channels
BO_resL = 4.9  # resistor on the blood outlet channel
WO_resL = 4.9  # resistor on wash side
tubL = 30  # length of tubing + flow meters
# Widths (um):
blood_w = 110  # blood channel widths
wash_w = 110  # wash channel widths
inj_w = 17  # injury widths
tubing_w = 254  # diameter of 0.01" ID tubing
# Heights (um):
inj_h = 10  # heights of injury channels
ch_h = 50


# Channel
def HCA_BC_Multiple_Inj_Counter(Qb=5.6, Qw=10,
                                lengths=[entL, entL, injL, inj_space, inj_space, injL, inj_space, inj_space, injL,
                                         inj_space, inj_space, injL, BO_resL, WO_resL, tubL, tubL],
                                widths=[blood_w, wash_w, inj_w, wash_w, blood_w, inj_w, blood_w, wash_w, inj_w,
                                        wash_w, blood_w, inj_w, blood_w, wash_w, tubing_w, tubing_w],
                                height=[ch_h, ch_h, inj_h, ch_h, ch_h, inj_h, ch_h, ch_h, inj_h, ch_h,
                                        ch_h, inj_h, ch_h, ch_h, ch_h, ch_h], Pb_out=0,
                                hematocrit=0.4, **kwargs):
    # ---- ------------Default Keyword Arguments--------------
    lengths = np.array(lengths)
    height = np.array(height)
    # channel widths in um
    widths = np.array(widths)
    # assign viscocities to channels. Using a farius-linquist approximations from
    # secomb, et al for channels that have 100% blood
    u = np.zeros(len(widths))
    for i in range(len(u)):
        # Channels 0,1,2,4,and 6 all have blood in them
        if i % 2 == 0 or i == 5 or i == 11:
            # Choose the smaller dimension of the channel (of w and h)
            # to Calculate the viscocity for
            if widths[i] < height[i]:
                d = widths[i]
            else:
                d = height[i]
            u[i] = mu_act(hematocrit, d)
        else:
            u[i] = 1
    # calculate resistances:
    R = np.zeros(len(lengths))
    for i in range(len(R)):
        # if the channel is tubing use the tubing resistance function, otherwise calculate treating as a rectangular duct
        if i == 14 or i == 15:
            R[i] = res_tubing(u[i], lengths[i], widths[i])
        else:
            R[i] = res_generalized_duct(u[i], lengths[i], widths[i], height[i])
    # make linear system of equations for pressure drop and flowrate equations
    # Coeffecient matrix
    # Either dP[i]-Q[i]R[i]=0 or flow in and out of a node sum to zero
    # Index and Corresponding variable below
    # 0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29
    # P0	P1	P2	P3	P4	P5	P6	P7	P8	P9	P10	P11	P12	P13	Q0	Q1	Q2	Q3	Q4	Q5	Q6	Q7	Q8	Q9	Q10	Q11	Q12	Q13	Q14	Q15
    #

    total_variables = 30
    A = np.zeros([total_variables, total_variables])
    A[0, 0] = 1;
    A[0, 2] = -1;
    A[0, 14] = -R[0]  # P0-P1-Q0R0=0
    A[1, 1] = 1;
    A[1, 3] = -1;
    A[1, 15] = -R[1]  # P1-P3-Q3R3=0
    A[2, 2] = 1;
    A[2, 3] = -1;
    A[2, 16] = -R[2]  # P2-P3-Q3R3=0
    A[3, 14] = 1;
    A[3, 16] = -1;
    A[3, 18] = -1;  # Q0-Q2-Q4=0
    A[4, 16] = 1;
    A[4, 23] = 1;
    A[4, 27] = -1  # Q2+Q9-Q13=0 ------
    A[5, 2] = 1;
    A[5, 4] = -1;
    A[5, 18] = -R[4]  # P2-P4=Q2R4=0
    A[6, 3] = 1;
    A[6, 5] = -1;
    A[6, 17] = -R[3]  # P3-P5-Q3R3=0
    A[7, 4] = 1;
    A[7, 5] = -1;
    A[7, 19] = -R[5]  # P4-P5-Q5R5=0
    A[8, 18] = 1;
    A[8, 19] = -1;
    A[8, 20] = -1  # Q4-Q5-Q6=0
    A[9, 21] = 1;
    A[9, 19] = 1;
    A[9, 23] = -1  # Q7-Q5-Q9=0 -----
    A[10, 4] = 1;
    A[10, 6] = -1;
    A[10, 20] = -R[6]  # P4-P6-Q6R6
    A[11, 5] = 1;
    A[11, 7] = -1;
    A[11, 21] = -R[7]
    A[12, 20] = 1;
    A[12, 22] = -1;
    A[12, 24] = -1  # Q6-Q8-Q10=0
    A[13, 17] = 1;
    A[13, 21] = -1;
    A[13, 22] = 1  # Q3-Q7+Q8=0 -----
    A[14, 6] = 1;
    A[14, 7] = -1;
    A[14, 22] = -R[8]  # P6-P7-Q8R8=0
    A[15, 6] = 1;
    A[15, 8] = -1;
    A[15, 24] = -R[10]  # P6-P8-Q10R10=0
    A[16, 7] = 1;
    A[16, 9] = -1;
    A[16, 23] = -R[9]  # P7-P9-Q9R9=0
    A[17, 8] = 1;
    A[17, 9] = -1;
    A[17, 25] = -R[11]  # P8-P9-Q11R11=0
    A[18, 24] = 1;
    A[18, 25] = -1;
    A[18, 26] = -1  # Q10-Q11-Q12=0
    A[19, 15] = 1;
    A[19, 17] = -1;
    A[19, 25] = 1  # Q1-Q3+Q11=0 -----
    A[20, 8] = 1;
    A[20, 10] = -1;
    A[20, 26] = -R[12]  # P8-P10-Q12R12=0
    A[21, 9] = 1;
    A[21, 11] = -1;
    A[21, 27] = -R[13]  # P9-P11-Q13R13=0
    A[22, 26] = 1;
    A[22, 28] = -1;  # Q12-Q14=0
    A[23, 27] = 1;
    A[23, 29] = -1  # Q13-Q15=0
    A[24, 10] = 1;
    A[24, 12] = -1;
    A[24, 28] = -R[14]  # P10-P12-Q14R14=0
    A[25, 11] = 1;
    A[25, 13] = -1;
    A[25, 29] = -R[15]  # P11-P13-Q15R15=0
    A[26, 12] = 1  # P12=Pb_out
    A[27, 13] = 1  # P13=0
    A[28, 15] = 1  # Q1=QW
    A[29, 14] = 1  # Q0=QB
    # answer matrix for coeffecient matrix
    B = np.zeros([total_variables])
    B[26] = Pb_out
    B[28] = Qw
    B[29] = Qb
    # Solve for matrix
    x = linalg.solve(A, B)
    # make values less than 1E-6 0
    for i in range(len(x)):
        if np.absolute(x[i]) < float(1E-6):
            x[i] = 0
    # extract the pressures and flowrates from the matrix
    p = x[:14]
    flows = x[14:]
    # get the pressure in and pressure out for the dataframe storage
    p_in = [p[0], p[1], p[2], p[3], p[2], p[4], p[4], p[5], p[6], p[7], p[6], p[8], p[8], p[9], p[10], p[11]]
    p_in = np.array(p_in)
    p_out = [p[2], p[3], p[3], p[5], p[4], p[5], p[6], p[7], p[7], p[9], p[8], p[9], p[10], p[11], p[12], p[13]]
    # calculate average wall shear rate for channels
    shear = np.zeros(len(flows))
    for i in range(len(shear)):
        shear[i] = shear_generalized_rectangle(flows[i], widths[i], height[i])

    channel_name = ['Blood In', 'Wash In', 'Inj 1', 'Wash 2', 'Blood 2', 'Inj 2', 'Blood 3', 'Wash 3', 'Inj 3',
                    'Wash 4', 'Blood 4', 'Inj 4', 'Blood Out Res', 'Wash Out Res', 'Blood Out Tubing', 'Wash Out Tubing'
                    ]
    dp = p_in - p_out
    # store results in dataframe to output
    output_df = pd.DataFrame()
    output_df['Flowrate (uL/min)'] = flows
    output_df['p_in (kPa)'] = p_in
    output_df['p_out (kPa)'] = p_out
    output_df['dP (kPa)'] = dp
    output_df['Channel'] = channel_name
    output_df['Shear (s^-1)'] = shear
    output_df['Viscocity (cP)'] = u
    output_df['Resistance (kPa-min/uL)'] = R
    output_df['Length (mm)'] = lengths
    output_df['Width (um)'] = widths
    output_df['Height (um)'] = height
    return output_df


# ---------------- Auxillary Functions Used in This Code --------------------
# ------function for calculating the resistance (kPa-s/uL) for generalized rect
#   channel
#   u(cP),L(mm),w(um),h(um)
def res_channel(u, L, w, h):
    L = L * 1000  # convert L from mm to um
    u = u
    w = w
    h = h
    if w < h:
        swap = w
        w = h
        h = swap
    R = 12 * u * L / w / (h ** 3) / (1 - 0.63 * h / w) * 1000 / 60
    return R


def res_generalized_duct(u, L, w, h):
    # source: Lehmann, M., Wallbank, A. M., Dennis, K. A., Wufsus, A. R., Davis, K. M., Rana, K., & Neeves, K. B. (2015).
    # On-chip recalcification of citrated whole blood using a microfluidic herringbone mixer. Biomicrofluidics, 9(6).
    # https://doi.org/10.1063/1.4935863
    L = L * 1000  # convert L from mm to um
    u = u
    w = w
    h = h
    if h < w:
        b = h
        a = w
    else:
        b = w
        a = h
    error = 100
    sum_loop = 0
    count = 1
    while error > 1e-3:
        sum_before = sum_loop
        sum_loop += np.tanh(count * np.pi * a / 2 / b) / (count ** 5)
        if count > 1:
            error = np.absolute(sum_loop - sum_before) / sum_before * 100
        count += 1
    denominator = 1 - 192 * b / (np.pi ** 5) / w * sum_loop
    resistance = 12 * u * L / a / (b ** 3) / denominator * 1000 / 60
    return resistance


# ------function for calculating the resistance (kPa-min/uL) for cylindrical tube
# u(cP),L(mm),w(um),h(um)
def res_tubing(u, l, d):
    r = d / 2
    l = l * 1000
    res = 8 * u * l / np.pi / r ** 4 * 1000 / 60
    return res


# ----- function for calculating shear for large aspect ratios (w>>h)
# inputs:
#   Q=flowrate (uL/min)
#   w = channel width (um)
#   h = channel height (um)
def simple_shear_rate(Q, w, h):
    # units from input units to SI
    Q = Q * 1e-9 / 60
    w = w * 1e-6
    h = h * 1e-6
    if w < h:
        swap = w
        w = h
        h = swap
    shear = 6 * Q / w / (h ** 2)
    return shear


# ----- function from Secomb that calculates the viscocity of blood in a channel
#       for a given hematocrit and shear rate
def mu_rel(h, d):
    C = (0.8 + np.exp(-.75 * d)) * (-1 + 1.0 / (1 + 10e-11 * (np.power(d, 12)))) + 1.0 / (
            1 + 10e-11 * (np.power(d, 12)))
    mu_fourtyfive = 220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-.06 * (np.power(d, .645)))
    mu = 1 + (mu_fourtyfive - 1) * ((1 - h) ** C - 1) / (np.power((1 - 0.45), C) - 1)
    return mu


# Function that takes the relative viscocity and nonnormalizes it for calculation purposes
def mu_act(h, d):
    act = 1.8 * mu_rel(h, d)
    return act


def shear_generalized_rectangle(Q, w, h):
    # ****calculating the average wall shear rate of a generalized rectangular
    # duct****
    # --------
    # Source: Predicting Non-newtonian Flow Behaviour of Ducts of Unusual
    # Cross section. Chester Miller.
    # https://pubs.acs.org/doi/pdf/10.1021/i160044a015
    # --------

    # units from input units to SI
    Q = Q * 1e-9 / 60
    w = w * 1e-6
    h = h * 1e-6
    # make a the largest dimension of duct and b smallest
    if h < w:
        a = w
        b = h
    else:
        a = h
        b = w
    r = b / a
    # calculate lambda factor
    denominator = ((1 - 0.351 * r) * (1 + r)) ** 2
    lamb = 24 / denominator
    # Hydraulic diameter
    Dh = 2 * a * b / (a + b)
    # Cross sectional area
    A = a * b
    shear = Q * lamb / 2 / A / Dh
    return shear


