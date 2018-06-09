#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
MPC trajectory planner

Max Ahlberg
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import cvxpy
from cvxpy import *

# Parameter
ITERATION = 0
N = 14  # MPC Horizon
DT = 4  # delta s in [m]

MAX_ROAD_WIDTH = 3.0  # maximum road width [m]
MAX_PSI = math.pi / 4 #max heading error 45 degrees
MAX_SPEED = 150.0 / 3.6  # maximum speed [m/s]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]

MAX_C = 0.2 # maximum change in curvature [1/m^2]
MAX_J = 10.0 # maximum change in acceleration [m/s^3]


#Vehicle parameters
L = 3.0  # [m] wheel base of vehicle
lr = L*0.5 #[m]
lf = L*0.5 #[m]
Width = 2.0  # [m] Width of the vehicle

show_animation = True


class quinic_polynomial: #Skapar ett 5e grads polynom som beräknar position, velocity och acceleration

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # (position_xs, Velocity_xs, Acceleration_xs, P_xe, V_xe, A_xe, Time )

        # calc coefficient of quinic polynomial
        self.xs = xs 
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0 #Varför accelerationen delat med 2? -För att de skall bli rätt dimensioner i slutandan

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b) #Antagligen matris invers som löser a3,a4,a5. En form av jerk, jerk_derivata, jerk dubbelderivata

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
    def calc_point(self, t): # point on xs at t
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t): #speed in point at t
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t): # acceleration in point at t
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quinic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt


class MPC_path:

    def __init__(self):
        self.t = []
        #States
        self.d = [] #Deviation from centerline
        self.psi = [] #Yaw angle of car
        self.v = [] #Speed
        self.K = [] #Curvature of car trajectory
        self.a = [] #Acceleration
        #from cost function
        self.scar = []
        #Inputs
        self.C = []
        self.J = []
        #Track
        self.s = [] #Position along the S-coordinate in frenet frame
        self.x = [] #Global
        self.y = [] #Global
        self.yaw = [] #Yaw angle of the Track
        self.c = []   #Curvature of the Track
        self.ds = []


class Vehcile_state:

    def __init__(self):
        self.t = []
        #States
        self.d = [] #Deviation from centerline
        self.psi = [] #Yaw angle of car
        self.v = [] #Speed
        self.K = [] #Curvature of car car (trajectory-curvature)
        self.a = [] #Acceleration
        #from cost function
        self.scar = []
        #Inputs
        self.C = []
        self.J = []
        #Track
        self.s = [] #Position along the S-coordinate in frenet frame
        self.x = [] #Global
        self.y = [] #Global
        self.yaw = [] #Yaw angle of the Track
        self.c = []   #Curvature of the Track
        self.ds = []


class Track_info:

    def __init__(self):
        # Track
        self.s = []  # Position along the S-coordinate in frenet frame
        self.x = []  # Global
        self.y = []  # Global
        self.yaw = []  # Yaw angle of the Track
        self.c = []  # Curvature of the Track
        self.ds = []



def MPC_calc(csp, mpc_est, ITERATION, track_info):
    mpcp = MPC_path()


    if ITERATION == 0:
        # track info
        s_track = track_info.s
        c = track_info.c
        yaw = track_info.yaw

        #initial position x_0
        scar = mpc_est.scar #Starta på s=0 måste vara ok!
        # optimal cost deltas
        s = mpc_est.s[0:(N)+1]

        #states
        psi_bar = mpc_est.e_psi[0:(N)+1]
        psi = mpc_est.e_psi[0]
        d_bar = mpc_est.d[0:(N)+1]
        d = mpc_est.d[0]
        v_bar = mpc_est.v[0:(N)+1]
        v = mpc_est.v[0]
        K_bar = mpc_est.K[0:(N)+1]
        K = K_bar[0]
        a_bar = mpc_est.a[0:(N)+1]
        a = a_bar[0]

        #inputs
        C_bar = mpc_est.C[0:(N-1)+1]
        C = K_bar[0]
        J_bar = mpc_est.J[0:(N-1)+1]
        J = J_bar[0]


        #HIT ÄR ALLT BRA!

    else:
        #track info
        s_track = track_info.s
        c = track_info.c
        yaw = track_info.yaw

        #states only the actual state needed to predict future x and u's
        scar = mpc_est.scar

        psi = mpc_est.psi[1]
        d = mpc_est.d[1]
        v = mpc_est.v[1]
        K = mpc_est.K[1]
        a = mpc_est.a[1]

        C = mpc_est.C[1]
        J = mpc_est.J[1]

        #The reference/privious prediction vectors
        s = mpc_est.s[0:(N)+1]

        psi_bar = mpc_est.psi[1:(N)+1]
        d_bar = mpc_est.d[1:(N)+1]
        v_bar = mpc_est.v[1:(N)+1]
        K_bar = mpc_est.K[1:(N)+1]
        a_bar = mpc_est.a[1:(N)+1]

        C_bar = mpc_est.C[1:(N-1)+1] #skall dessa framåt?
        J_bar = mpc_est.J[1:(N-1)+1]

    print "REFERENS psi:", psi_bar
    print "REFERENS d:", d_bar
    print "REFERENS v:", v_bar
    print "REFERENS K:", K_bar
    print "REFERENS a:", a_bar
    print "REFERENS s:", s



    X_0 = [d, psi, v, K, a]
    U_0 = [C, J]
    x_bar_vec = np.matrix([d_bar,
                        psi_bar,
                        v_bar,
                        K_bar,
                        a_bar])
    u_bar_vec = np.matrix([C_bar,
                       J_bar])
    n = 5  # States x
    m = 2  # Control signals u
    print "X_0:", X_0
    print "U_0:", U_0

    x = cvxpy.Variable(n, N)
    u = cvxpy.Variable(m, N-1)
    slack = cvxpy.Variable(n, N)
    slackC = cvxpy.Variable(1,1)
    slackJ = cvxpy.Variable(1,1)

    cost_matrix = np.eye(n,n)
    Q_slack = cost_matrix * 1
    Q_inputs = 1
    cost = 0.0
    constr = []
    c_bar = []


    for t in range(N-1): #Detta är som att köra MPCn fär alla N states, är de korrekt?
        x_bar = x_bar_vec[:, t]
        u_bar = u_bar_vec[:, t]
        s_aprox = s[t] #Mpc estimerade s punkterna
        s_point = s_aprox
        idx = (np.abs(np.asarray(s_track) - s_point)).argmin()
        c_bar.append(c[idx])
        yaw_prime = (yaw[idx+1]-yaw[idx])/0.05 #ds = 0.05 from csp. I teorin samma som c_bar
        #print "is yaw prime -1/R?:", yaw_prime, "c_bar:", c_bar[-1]
        #TESTAT byta ut K_bar mot yaw_prime för att inte vara beroende av mina gissningar.

        F = np.matrix([[(1-d_bar[t]*c_bar[-1])*math.tan(psi_bar[t])],
                       [((1-d_bar[t]*c_bar[-1])*K_bar[t])/(math.cos(psi_bar[t])) - yaw_prime],
                       [((1-d_bar[t]*c_bar[-1])*a_bar[t])/(v_bar[t]*math.cos(psi_bar[t]))],
                       [((1-d_bar[t]*c_bar[-1])*C_bar[t])/(v_bar[t]*math.cos(psi_bar[t]))],
                       [((1-d_bar[t]*c_bar[-1])*J_bar[t])/(v_bar[t]*math.cos(psi_bar[t]))]])

        A = np.matrix([[-c_bar[-1]*math.tan(psi_bar[t]),
                        ((1-d_bar[t]*c_bar[-1]))/(math.cos(psi_bar[t])**2),
                        0,
                        0,
                        0],
                        [-(c_bar[-1]*K_bar[t])/(math.cos(psi_bar[t])),
                         (((1 - d_bar[t] * c_bar[-1]) * K_bar[t])*math.tan(psi_bar[t])) / (math.cos(psi_bar[t])),
                         0,
                         (1-d_bar[t]*c_bar[-1])/math.cos(psi_bar[t]),
                         0],
                       [-(c_bar[-1]*a_bar[t])/(v_bar[t]*math.cos(psi_bar[t])),
                        ((1-d_bar[t]*c_bar[-1])*a_bar[t]*math.tan(psi_bar[t]))/(v_bar[t]*math.cos(psi_bar[t])),
                        -((1-d_bar[t]*c_bar[-1])*a_bar[t])/(v_bar[t]**2 *math.cos(psi_bar[t])),
                        0,
                        (1-d_bar[t]*c_bar[-1])/(v_bar[t]*math.cos(psi_bar[t]))],
                       [-(c_bar[t]*C_bar[t])/(v_bar[t]*math.cos(psi_bar[t])),
                        ((1-d_bar[t]*c_bar[-1])*C_bar[t]*math.tan(psi_bar[t]))/(v_bar[t]*math.cos(psi_bar[t])),
                        -((1-d_bar[t]*c_bar[-1])*C_bar[t])/(v_bar[t]**2 * math.cos(psi_bar[t])),
                        0,
                        0],
                       [-(c_bar[t]*J_bar[t])/(v_bar[t]*math.cos(psi_bar[t])),
                        ((1 - d_bar[t] * c_bar[-1]) * J_bar[t] * math.tan(psi_bar[t])) / (v_bar[t] * math.cos(psi_bar[t])),
                        -((1 - d_bar[t] * c_bar[-1]) * J_bar[t]) / (v_bar[t] ** 2 * math.cos(psi_bar[t])),
                        0,
                        0]])

        invert = is_invertible(A)
        print "Invertable?: ", invert

        #Ad = np.eye(n, n) + (DT*A)

        B = np.matrix([[0, 0],
                       [0, 0],
                       [0, 0],
                       [(1-d_bar[t]*c_bar[-1])/(v_bar[t] * math.cos(psi_bar[t])), 0],
                       [0, (1-d_bar[t]*c_bar[-1])/(v_bar[t] * math.cos(psi_bar[t]))]])

        M_matrix = np.block([[A, B],
                             [np.eye(m,n)*0, np.eye(m,m)*0]])
        M_matrix = DT *M_matrix
        M = scipy.linalg.expm(M_matrix)
        Ad = M[:n, :n]
        Bd = M[:n, n:]

        #Pedro + Max cost function
        cost += -(c_bar[-1])/(v_bar[t])*x[0, t]
        cost += (1 - d_bar[t]*c_bar[-1])/(2*v_bar[t])*x[1, t]**2
        cost += -((1-d_bar[t]*c_bar[-1]))/(v_bar[t]**2)*x[2, t]
        cost += sum_squares(Q_slack*slack[:,t])
        cost += slackC**2 * Q_inputs
        cost += slackJ**2 * Q_inputs


        constr += [x[:, t + 1] == Ad * x[:, t] + Bd * u[:, t]+ F - A*x_bar - B*u_bar  ]#se till att implementera x_bar rätt

        constr += [x[0, t+1] <= MAX_ROAD_WIDTH + slack[0,t]]  #Lateral Deviation e_y
        constr += [x[0, t+1] >= -MAX_ROAD_WIDTH - slack[0,t]]
        constr += [x[1, t+1] <= MAX_PSI + slack[1,t]]  #Angular deviation e_psi
        constr += [x[1, t+1] >= -MAX_PSI - slack[1,t]]
        constr += [x[2, t+1] <= MAX_SPEED + slack[2,t]] #Velocity v_x
        constr += [x[2, t+1] >= 0 - slack[2,t]]
        constr += [x[3, t+1] <= MAX_CURVATURE + slack[3,t]]  #Curvature K = tan(delta)/l
        constr += [x[3, t+1] >= -MAX_CURVATURE - slack[3,t]]
        constr += [x[4, t+1] <= MAX_ACCEL + slack[4,t]] #Acceleration a_x
        constr += [x[4, t+1] >= -MAX_ACCEL - slack[4,t]]


        constr += [u[0, t] <=  MAX_C + slackC]  #Sharpness C
        constr += [u[0, t] >= -MAX_C - slackC]
        constr += [u[1, t] <= MAX_J + slackJ]  #Longitudional Jerk
        constr += [u[1, t] >= -MAX_J - slackJ]

        #slack variables
        constr += [slack[:,t] >= 0]
        constr += [slackC >= 0]
        constr += [slackJ >= 0]

        #print("Constr is DCP:", (x[:, t + 1] == (A * x[:, t] + B * u[:, t])).is_dcp())
        '''
        print("Constr is DCP road with   :", (x[0, t] <= MAX_ROAD_WIDTH).is_dcp(), "and: ", (x[0, t] >= -MAX_ROAD_WIDTH).is_dcp())
        print("Constr is DCP angle error :", (x[1, t] <= MAX_PSI).is_dcp(), "and: ", (x[1, t] >= -math.pi/2).is_dcp())
        print("Constr is DCP velocity    :", (x[2, t] <= MAX_SPEED).is_dcp(), "and: ", (x[2, t] >= 0).is_dcp())
        print("Constr is DCP Kurvature   :", (x[3, t] <= MAX_CURVATURE).is_dcp(), "and: ", (x[3, t] >= -MAX_CURVATURE).is_dcp())
        print("Constr is DCP Acceleration:", (x[4, t] <= MAX_ACCEL).is_dcp(), "and: ", (x[4, t] >= -MAX_ACCEL).is_dcp())
        '''


    constr += [x[:, 0] == X_0]
    #constr += [u[:, 0] == U_0]
    print("Constr is DCP:", (x[:, 0] == X_0).is_dcp())

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    print("prob is DCP:", prob.is_dcp())
    print("Cost is DCP:", cost.is_dcp())

    print("curvature of x:", x.curvature)
    print("curvature of u:", u.curvature)
    print("curvature of cost:", cost.curvature)
    prob.solve(verbose=False)
    print "Status:", prob.status
    print "Optimal value with ECOS:", prob.value


    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        #The outputs from the MPC is the real states
        mpcp.psi, mpcp.d, mpcp.v, mpcp.K, mpcp.a, mpcp.s = [], [], [], [], [], []


        mpcp.d = x.value[0, :].tolist()
        mpcp.d = mpcp.d[0]
        mpcp.psi = x.value[1, :].tolist() #De riktiga statsen
        mpcp.psi = mpcp.psi[0]
        mpcp.v = x.value[2, :].tolist()
        mpcp.v = mpcp.v[0]
        mpcp.K = x.value[3, :].tolist()
        mpcp.K = mpcp.K[0]
        mpcp.a = x.value[4, :].tolist()
        mpcp.a = mpcp.a[0]

        mpcp.C = u.value[0, :].tolist()
        mpcp.C = mpcp.C[0]
        mpcp.J = u.value[1, :].tolist()
        mpcp.J = mpcp.J[0]
        #beräkningen av s är fel! Ända felet kvar som jag ser nu

        #lägger till de interpolerade sista statet


        mpcp.d.append(mpcp.d[-1])
        mpcp.psi.append(mpcp.psi[-1])
        mpcp.v.append(mpcp.v[-1])
        mpcp.K.append(mpcp.K[-1])
        mpcp.a.append(mpcp.a[-1])

        mpcp.C.append(mpcp.C[-1])
        mpcp.J.append(mpcp.J[-1])

        #gör en egen loop för att få rätt S
        mpcp.s.append(scar)
        for l in range(N-2):
            #mpcp.s.append(scar + ((DT/mpcp.v[l])*mpcp.v[l]*math.cos(mpcp.psi[l]))/(1 - mpcp.d[l]*c_bar[l]))
            scar = mpcp.s[-1]
        #mpcp.s.append(scar + ((DT/mpcp.v[-1]) * mpcp.v[-1] * math.cos(mpcp.psi[-1])) / (1 - mpcp.d[-1] * c_bar[-1]))
        #mpcp.s.append(scar + ((DT/mpcp.v[-1]) * mpcp.v[-1] * math.cos(mpcp.psi[-1])) / (1 - mpcp.d[-1] * c_bar[-1]))
        mpcp.s = s[1:]
        mpcp.s.append(mpcp.s[-1] + DT)
        mpcp.scar = mpcp.s[1]
        #Mycket verkar ok till hit!
        print "mpc solution psi:", mpcp.psi
        print "mpc solution d:", mpcp.d
        print "mpc solution v:", mpcp.v
        print "mpc solution K:", mpcp.K
        print "mpc solution a:", mpcp.a
        print "mpc solution s:", mpcp.s
        sd = slack.value[0,:]
        spsi = slack.value[1,:]
        sv = slack.value[2,:]
        sK = slack.value[3,:]
        sa = slack.value[4,:]

        slackC = slackC.value
        slackJ = slackJ.value
        print "The Slack d variable is:", sd
        print "The Slack psi variable is:", spsi
        print "The Slack v variable is:", sv
        print "The Slack K variable is:", sK
        print "The Slack a variable is:", sa

        print "The Slack C variable is:", slackC
        print "The Slack J variable is:", slackJ


    return mpcp

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

'''
def calc_global_paths(mpc_path, csp): #Beräknar trajectories i globala koordinatsystemet från S och D i Frenet
    #Gör så att denna går från MPC output till plot bart

    for fp in mpc_path:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])#beräknar yaw Angle i varje punkt som derivatan av kurvan
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0) #punkten x + närliggande katet 
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)# punkten y + frånliggande katet
            fp.x.append(fx) #lägger alla globala x coordinater i en lista
            fp.y.append(fy) #lägger alla globala y coordinater i en lista

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx)) #Yaw angle -> derivatan på kurvan
            fp.ds.append(math.sqrt(dx**2 + dy**2)) # längden på tangent vectorn

        fp.yaw.append(fp.yaw[-1]) #lägger till sista yaw'n en gång till, antagligen för att få samma längd på vektorer
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i]) #beräknar förändringen i yaw per steg längs sträckan.

    return fplist
'''
'''
def global_vehicle_simulator(mpc_path):

    states = Vehcile_state()
    print "type of mpc_path:", type(mpc_path)
    #Tar bara nästa steg i MPCn som de sanna nästa statet
    #Updatera .s_car!!
    # from cost function
    states.scar = mpc_path.scar
    states.s = np.squeeze(np.asarray((mpc_path.s)))
    # States
    states.psi = np.squeeze(np.asarray(mpc_path.psi))  # Yaw angle of car
    states.psi = states.psi[0]
    states.d = np.squeeze(np.asarray(mpc_path.d))  # Deviation from centerline
    states.d = states.d[0]
    states.v = np.squeeze(np.asarray(mpc_path.v))  # Speed
    states.v = states.v[0]
    # Inputs
    states.K = np.squeeze(np.asarray(mpc_path.K))  # Curvature of car car (trajectory-curvature)
    states.K = states.K[0]
    states.a = np.squeeze(np.asarray(mpc_path.a))  # Acceleration
    states.a = states.a[0]

    vehicle_state = states

    return vehicle_state
'''

def frenet_optimal_planning(csp, mpc_est, ITERATION, track_info):

    mpc_path = MPC_calc(csp, mpc_est, ITERATION, track_info)

    #mpc_path_global = calc_global_paths(mpc_path, csp)#De beräknade trajektorerna görs om från Frenet till globala

    #vehicle_state = global_vehicle_simulator(mpc_path)

    return mpc_path


def generate_target_course(x, y):#tar manuelt inmatate coordinater och skapar ett polynom som blir referens!
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.05)
    d = np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, 0.05) #Skapa 0.1 tät vinkelrät linje
    s_len = s.size
    d_len = d.size
    LUTs = np.zeros((s_len, d_len))
    LUTd = np.zeros((s_len, d_len))
    LUTx = np.zeros((s_len, d_len))
    LUTy = np.zeros((s_len, d_len))

    rs, rx, ry, ryaw, rk = [], [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s) #i_s  = incremental s ->här kan vi göra visualitionsconstraintsen
        rx.append(ix)#Ref x
        ry.append(iy)#Ref y
        ryaw.append(csp.calc_yaw(i_s))#Ref yaw
        rk.append(csp.calc_curvature(i_s))#Ref curvature
        rs.append(i_s)

    LTs, LTd, LTx, LTy, refx, refy, refyaw = [], [], [], [], [], [], []
    s_count, d_count = -1, -1
    for i_ss in s:
        s_count = s_count + 1
        LTs = i_ss
        refx, refy = csp.calc_position(i_ss)
        refyaw = csp.calc_yaw(i_ss)
        for i_dd in d:
            if i_dd == -MAX_ROAD_WIDTH:
                d_count = -1
            d_count = d_count + 1
            LTd = -i_dd
            LTx = refx + i_dd*math.sin(refyaw)
            LTy = refy - i_dd*math.cos(refyaw)
            LUTs[s_count, d_count] = LTs
            LUTd[s_count, d_count] = LTd
            LUTx[s_count, d_count] = LTx
            LUTy[s_count, d_count] = LTy

    plt.plot(LUTx[:,:], LUTy[:,:], LUTx[2000,2], LUTy[2000,2], 'D')
    plt.plot(LUTs[:,:], -3*MAX_ROAD_WIDTH + LUTd[:,:], LUTs[2000, 2], -3*MAX_ROAD_WIDTH + LUTd[2000, 2], 'D')
    plt.grid(True)
    plt.show()


    return rs, rx, ry, ryaw, rk, csp, LUTs, LUTd, LUTx, LUTy




def main():
    print(__file__ + " start!!")


    # way points

    #wx = [0.0, 1000.0]
    #wy = [0.0, 0.0]

    #wx = [0.0, 10, 50, 70, 100, 105, 110, 115, 120, 120, 120, 120, 130, 140, 160]
    #wy = [0, 0, 0,  0,   0,  0, 0, 0, 25, 35, 36, 55, 55, 55, 55]

    wx = [-50, 0.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0] #, 500.0, 800.0, 1000.0]
    wy = [0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  5.0,  30.0,  30.0,  5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #, 0.0, 0.0, 0.0]

    #wx = [0.0, 10, 20.0,  30,  40,  80,  85,  90, 95.0, 100, 110, 115, 110, 100, 95, 90, 85, 80, 40, 30, 20, 10,  0, -1000]
    #wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 5.0,  15,  25,  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]

    #wx = [-20.0, -40.0, -70.0, -100.0, -120.0,  -140.0,  -150.0,   -160.0, -180.0, -200.0, -180.0, -160.0, -150.0, -140.0, -130.0, -120.0, -90.0, -60.0, -40.0, 0.0, 5.0, 0.0, -15.0, -20.0]
    #wy = [0.0, 0.0,  5.0,  0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 20.0, 40.0, 40.0, 40.0, 45.0, 40.0, 35.0, 40.0, 40.0, 40.0, 40.0, 20.0, 0.0, 0.0, 0.0]

    ts, tx, ty, tyaw, tc, csp, LUTs, LUTd, LUTx, LUTy = generate_target_course(wx, wy)


    # initial state, giving the car a initial position
    track_info = Track_info()
    mpc_s = MPC_path()

    mpc_s.scar = ts[0]
    # States
    mpc_s.d = [0.0]      # Deviation from centerline
    mpc_s.d += N * mpc_s.d  # Tot [0-N,N]
    mpc_s.e_psi = [0.0]  # Yaw angle deviation of car compared to track
    mpc_s.e_psi += N * mpc_s.e_psi  # Tot [0-N,N]
    mpc_s.v = []
    mpc_s.K = [0.0]   # Curvature of car car(global) (trajectory-curvature, estimated to go straight ahead)
    mpc_s.a = [0.0]  # Acceleration
    mpc_s.K += N * mpc_s.K # Tot [0-(N-1),(N-1)]
    mpc_s.a += N * mpc_s.a # Tot [0-(N-1),(N-1)]

    mpc_s.s = []
    mpc_s.s.append(0)
    for i in range(N+1):
        mpc_s.v.append(40.0+(i*1)*0)   # Speed  # Tot [0-N,N]
    for i in range(N):
        mpc_s.s.append(mpc_s.s[-1] + DT)  # Tot [0-N,N]
    # Inputs
    mpc_s.C = [0.0]  # Sharpness (trajectory-curvature change per change in , estimated to make no change)
    mpc_s.C += (N-1) * mpc_s.C # Tot [0-(N-1),(N-1)]
    mpc_s.J = [0.0]  # Jerk
    mpc_s.J += (N-1) * mpc_s.J# Tot [0-(N-1),(N-1)]

    mpc_est = mpc_s
    initial = mpc_s

    #Track
    track_info.s, track_info.c, track_info.yaw, track_info.x, track_info.y = ts, tc, tyaw, tx, ty

    displacement, travel, x_path, y_path, steering, v, acceleration = [], [], [], [], [], [], []

    area = 100
    for i in range(500):#Antalet gånger koden körs. Bryts när målet är nått!! Detta blir Recursive delen i MPC'n
        ITERATION = i
        print "ITERATION: ", ITERATION
        #saving all values:
        displacement.append(mpc_est.d[1])
        travel.append(mpc_est.s[1])
        acceleration.append(mpc_est.a[1])
        x, y = [], []
        for i in range(N):
            idx, x_coord, y_coord, ix, iy = [], [], [], [], []
            X = np.sqrt(np.square(LUTs - mpc_est.s[i]) + np.square(LUTd - mpc_est.d[i]))
            idx = np.where(X == X.min())
            ix = np.asscalar(idx[0][0])
            iy = np.asscalar(idx[1][0])
            x_coord = LUTx[ix, iy]
            y_coord = LUTy[ix, iy]
            x.append(x_coord)
            y.append(y_coord)

        x_path.append(x[0])
        y_path.append(y[0])
        steering.append(math.atan(mpc_est.K[1]/L))
        v.append(mpc_est.v[1])

        mpc_path = frenet_optimal_planning(csp, mpc_est, ITERATION, track_info) #The magic



        '''
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break
        '''
        if show_animation:
            plt.cla()
            '''
            plt.figure(1)
            plt.subplot(231)
            plt.plot(mpc_est.s[:], mpc_est.a, '-ok')
            plt.title('Acceleration predictions')
            plt.ylim(-MAX_ACCEL - 5, MAX_ACCEL + 5)

            plt.subplot(232)
            plt.plot(mpc_path.s, mpc_path.d, '-or')
            plt.title('Displacement predictions')
            plt.ylim(-3, 3)

            plt.subplot(233)
            plt.plot(mpc_est.s, mpc_est.v, '-og')
            plt.title('Velecity predictions')

            plt.subplot(234)
            plt.plot(travel, acceleration, 'k')
            plt.title('Acceleration')
            plt.ylim(-MAX_ACCEL - 5, MAX_ACCEL + 5)

            plt.subplot(235)
            plt.plot(travel, displacement, 'r', LUTs[:,0], LUTd[:,0], 'b', LUTs[:,-1], LUTd[:,-1], 'y')
            plt.title('Displacement')


            plt.subplot(236)
            plt.plot(travel, v, 'g')
            plt.title('Velocity')
            '''

            plt.savefig('/home/maxahlberg/Pictures/my_new_fig1.png')
            plt.figure(1)
            plt.plot(x_path, y_path, LUTx[:, 0], LUTy[:, 0], 'b', LUTx[:, -1], LUTy[:, -1], 'y', x, y, '-or')
            plt.title("Iteration: " + '{:.2f}'.format(ITERATION) +
                      " v: " + '{:.2f}'.format(mpc_est.v[0])
                      + " K[-1]: " + '{:.2f}'.format(mpc_est.K[-1]))
            #plt.xlim(-10, 120)
            #plt.ylim(-10, 50)
            plt.savefig('/home/maxahlberg/Pictures/my_new_fig2.png')


            plt.figure(2)
            plt.subplot(231)
            plt.plot(mpc_path.s, mpc_path.d, '-or', mpc_path.s, mpc_est.d, '.g')
            plt.title('Displacement predictions')

            plt.subplot(232)
            plt.plot(mpc_path.s, mpc_path.v, '-ok', mpc_path.s, mpc_est.v, '.g')
            plt.title('Velocity predictions')

            plt.subplot(233)
            #plt.plot(mpc_est.s, mpc_est.v, '-og')
            #plt.title('Velecity predictions')
            plt.plot(mpc_path.s, mpc_path.a, '-ok', mpc_path.s, mpc_est.a, '.g')
            plt.title('Acceleration predictions')

            plt.subplot(234)
            plt.plot(mpc_path.s, mpc_path.K, '-oy', mpc_path.s, mpc_est.K, '.g')
            plt.title('Curvature predictions')

            plt.subplot(235)
            plt.plot(mpc_path.s[:-1], mpc_path.C, '-oy', mpc_path.s[:-1], mpc_est.C, '.g')
            plt.title('Sharpness predictions')

            plt.subplot(236)
            plt.plot(mpc_path.s[:-1], mpc_path.J, '-ok', mpc_path.s[:-1], mpc_est.J, '.g')
            plt.title('Jerk')
            plt.show()

            # initial = vehicle_state
            mpc_est = mpc_path

            #plt.pause(0.001)

    print("Finish")
    if show_animation:
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
