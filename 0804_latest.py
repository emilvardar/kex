import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cp
import scipy as sp
from scipy import sparse
import warnings

# Vehicle parameters
PARA = 1  # Use to minimize the vehicle parameters
LENGTH = 4.5 / PARA  # [m]
WIDTH = 2.0 / PARA  # [m]
BACKTOWHEEL = 1.0 / PARA  # [m]
WHEEL_LEN = 0.3 / PARA  # [m]
WHEEL_WIDTH = 0.2 / PARA  # [m]5
TREAD = 0.7 / PARA  # [m]
WB = 2.5 / PARA  # [m]

# Road length
ROAD_LENGTH = 400 # [m]

# Initial acceleration
U_INIT = 0.0 # m/s^2

INIT_DIST = 10.0    # Initial distance between the cars

# Initial velocity
V_INIT = 90.0 / 3.6 # 90km/h -> 100/3.6 m/s

# Prediction horizon
PREDICTION_HORIZON = 30

# Constriants
SAFETY_DISTANCE = 2 + LENGTH # Albin --> Enligt Energy savings from an Eco-Cooperative Adaptive Cruise Control: a BEV platoon investigation så har de 2 meter som min
MAX_ACCELERATION = 5  # 5 m/s^2
MIN_ACCELERATION = -10 # 10 m/s^2 decleration
MAX_VEL = 120/3.6 # m/s Albin --> Enligt Energy savings from an Eco-Cooperative Adaptive Cruise Control: a BEV platoon investigation så har de max 144 och min 0 kph så vår är ju i så fall skitbra
MIN_VEL = 50/3.6 # m/s

# Max time
MAX_TIME = 25  # [s]

# Time step
DT = 0.2  # [s]

# Air drag coeff
RHO = 1.225 # kg/m^3
CD = 0.8   # Albin --> hittat från https://www.engineeringtoolbox.com/drag-coefficient-d_627.html
            # och från Energy savings from an Eco-Cooperative Adaptive Cruise Control: a BEV platoon investigation tillsammans
CD1 = 4.144 
CD2 = 7.538
# Albin --> Enligt Energy savings from an Eco-Cooperative Adaptive Cruise Control: a BEV platoon investigation så bör CD1 = 4.144 och CD2 = 7.538 för 94.84% R-square fit

AREA = 6 # m^2
WEIGTH_VEH = 40000.0 # kg
K = RHO*AREA*CD/WEIGTH_VEH

'''
'Albin' --> Diskutera med Albin
Svenska --> ställen jag har skrivit på svenska som vi kan tabort sist
'''

class VehicleState:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, v=0.0):
        self.x = x
        self.v = v


def update_states(veh_states, control_signals, drag_or_not):
    # updates states for all vehicles
    if drag_or_not == '0':
        # Ideal situation
        for i in range(NUM_CARS):
            veh_states[i].x = veh_states[i].x + veh_states[i].v * DT
            veh_states[i].v = veh_states[i].v + control_signals[-NUM_CARS + i]* DT
    else:
        # Air drag cond.
        for i in range(NUM_CARS):
            veh_states[i].x = veh_states[i].x + veh_states[i].v * DT 
            if i != 0:
                CD_distance = CD * (1 - (CD1 / (CD2 + (veh_states[i-1].x - veh_states[i].x))))
                acc_ad = 0.5 * RHO * AREA * (veh_states[i].v ** 2) * CD_distance / WEIGTH_VEH  # Air drag (in terms of acceleration) on vehicle i, which depends on the distance to preeceding vehicle
                veh_states[i].v = veh_states[i].v + (control_signals[-NUM_CARS + i] - acc_ad) * DT
    return veh_states
    

def mpc(veh_states, xref, split_car, last_ref_in_mpc, split_distance, drag_or_not, hard_split, x_last):
    """
    heavily inspired by https://www.cvxpy.org/tutorial/intro/index.html
    """
    # Even if air drag has been choosen the first mpc needs to calculate with A and B matrices for the ideal case
    Ad, Bd, Dd = create_matrices()  # Create A, B and D matrices for the ideal case
    Q, R = cost_matrices(split_car)
    QN = Q

    x0 = []
    for i in range(NUM_CARS-1):
        x0.append(veh_states[i].x - veh_states[i+1].x)
    for i in range(NUM_CARS):
        x0.append(veh_states[i].v)
    
    # Create two scalar optimization variables.
    u = cp.Variable((NUM_CARS-1, PREDICTION_HORIZON))
    x = cp.Variable((2*NUM_CARS-1, PREDICTION_HORIZON+1))
    
    constraints = [x[:,0] == x0] # First constraints needs to be the initial state
    cost = 0.0 # Define the cost function
    for k in range(PREDICTION_HORIZON):
        if type(x_last) != type(None) and drag_or_not == "1":   # The first mpc calculation doesn't have predicted values thus the ideal matrices are been used for the first iteration. Therefore the type is None in the first iteration but never after that.
            Ad, Bd, Dd = create_matrices_linear(x_last[:,k+1])
        cost += cp.quad_form(x[:,k] - xref, Q) + cp.quad_form(u[:,k], R)    # Add the cost function sum(x^TQx + u^TRu). Observe here that x(0) is included thus the formula in https://osqp.org/docs/examples/mpc.html is correct.
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k] + Dd]             # Add the system x(k+1) = Ad*x(k) + Bd*u(k) + Dd
        for i in range(NUM_CARS):                                           # The for loop is for defining safety distance for all cars
            constraints += [MIN_VEL <= x[NUM_CARS-1+i,k], x[NUM_CARS-1+i,k] <= MAX_VEL]  # Add the velocity constrain just on the velocity inputs in the state vector
            if i != NUM_CARS-1:
                constraints += [SAFETY_DISTANCE <= x[i,k]]  # Add distance constrain on just the distance inputs in the state vector
        if k >= (PREDICTION_HORIZON - last_ref_in_mpc) and hard_split:  # Svenska Den bör inte ligga i for loopen. Den gör om samma sak om och om igen fast den bara bör sätte hard constrainen 1 gång
            constraints += [[split_distance + LENGTH + 1.] <= x[split_car-1,k]] # +1 or +2 affects the overshoot
        constraints += [MIN_ACCELERATION <= u[:,k], u[:,k] <= MAX_ACCELERATION] # Constarins for the control signal
    cost += cp.quad_form(x[:,PREDICTION_HORIZON] - xref, QN) # Svenska Låt den vara kvar för rapportten. Det blir nog lättare att ha den kvar när vi skriver.
    
    # Form and solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    sol = prob.solve(solver=cp.ECOS)
    warnings.filterwarnings("ignore")
    #print(u.value)
    # x[:,0] is the states of the vehicles at this moment, i.e. when mpc is been calculated. x[:,1] is the state where the vehicles will be when we have done the calculations bellow, i.e. the time next mpc will be calculated 
    # u[:,0] is the value that the control signal should be at this moment. Thus we are applying that to the system. 
    return u[:,0].value, x.value  


def cost_matrices(split_car):
    # Cost matrices
    # R = 1.0 * sparse.eye(NUM_CARS-1)
    'Albin --> Har ändrat R till följande'
    r_vec = []
    for i in range(NUM_CARS-1):
        r_vec.append(1.0)
        if (i == split_car-2) or (i == split_car-1): # Make the preceeding splitting vehicle's control signal more important(-2) Make the splitting vehicle's control signal(-1) more important
            r_vec[-1] = 10
    R = sparse.diags(r_vec)
    
    q_vec = [0]*(2*NUM_CARS-1)
    for i in range(NUM_CARS):
        if i != NUM_CARS-1:
            q_vec[i] = 10.0
        q_vec[i+NUM_CARS-1] = 1.0
        if i == split_car-1:
            q_vec[i] = 4*10.0
            # q_vec[i-1] = 3*10.0 # Detta kan användas för att hålla avståndet framför splittens 2 bilar mer stabilt. Tex så att bil 1 och 2 inte kommer
            # för närma varandra när bil 2 och 3 ska splittas. 
    Q = sparse.diags(q_vec)
    return Q, R


def create_matrices():
    # Define dimentions for A and B matrices
    A = np.zeros((2*NUM_CARS-1,2*NUM_CARS-1))
    B = np.zeros((2*NUM_CARS-1,(NUM_CARS-1)))
    D = np.zeros(2*NUM_CARS-1)

    for i in range(NUM_CARS-1):
        A[i,NUM_CARS-1+i] = 1
        A[i,NUM_CARS+i] = -1

        B[NUM_CARS+i,i] = 1

    # Identity matrix
    I = sparse.eye(2*NUM_CARS-1)

    # Discretize A and B matrices
    Ad = (I + DT*A)
    Bd =  DT*B
    Dd = D
    return Ad, Bd, Dd


def create_matrices_linear(x_pred):
    # Define dimentions for A and B matrices
    A = np.zeros((2*NUM_CARS-1,2*NUM_CARS-1))
    B = np.zeros((2*NUM_CARS-1,(NUM_CARS-1)))
    D = np.zeros(2*NUM_CARS-1)

    for i in range(NUM_CARS-1):
        S, R, Q = deltax_velocity_dependence(x_pred,i)   # i=0 --> s2, r2, q2, [in comp. s1, r1, q1] i==1 --> s3, r3, q3

        A[i,NUM_CARS-1+i] = 1
        A[i,NUM_CARS+i] = -1
        A[NUM_CARS+i,i] = S
        A[NUM_CARS+i,NUM_CARS+i] = R

        B[NUM_CARS+i,i] = 1
        D[NUM_CARS+i] = Q

    # Identity matrix
    I = sparse.eye(2*NUM_CARS-1)

    # Discretize A and B matrices
    Ad = (I + DT*A)
    Bd =  DT*B
    Dd = DT*D
    return Ad, Bd, Dd


def deltax_velocity_dependence(x_pred,i):
    R = -K * x_pred[NUM_CARS+i] * (1 - CD1/(CD2 + x_pred[i])) # i+1 because when it is been sending to this func. the input is i e.g i=0 but we then need for the second vehicle which is i=1 actualy.
    S = -K/2 * (x_pred[NUM_CARS+i] ** 2) * (CD1/((CD2 + x_pred[i]) ** 2))
    Q = -K/2 * (x_pred[NUM_CARS+i]**2) * (x_pred[i] * (CD1/((CD2 + x_pred[i]) ** 2)) + (1 - CD1/(CD2 + x_pred[i]))) # Albin ändrade minuset i mitten till plus jag tror vi hade gjort det fel förut men kontrollera för säkerhetensskull
    return R, S, Q


def plot_car(x, y=0.0, yaw=0.0, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    '''Code from Author: Atsushi Sakai(@Atsushi_twi)'''
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def clear_and_draw_car(states):
    plt.cla()
    dl = 1  # course tick

    cx = []
    cy = []
    for i in range(ROAD_LENGTH):
        cx.append(i)
        cy.append(0)  # get the straight line

    plt.plot(cx, cy, "-r", label="course")
    for i in range(NUM_CARS):
        plot_car(states[i].x)  # plot the cars

    plt.axis([0, ROAD_LENGTH, -50, 50])
    plt.grid(True)
    plt.pause(0.0001)


def velocity_plotter(dt_list, v_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']

    v_list = [element * 3.6 for element in v_list]
    for i in range(NUM_CARS):
        plt.plot(dt_list, v_list[i::NUM_CARS], colors[i%7], label=('v' + str(i+1)))

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Time(s)")
    plt.ylabel("Velocity(kph)")
    plt.legend()
    plt.show()


def acceleration_plotter(dt_list, u_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']

    for i in range(NUM_CARS):
        plt.plot(dt_list, u_list[i::NUM_CARS], colors[i%7], label=('a' + str(i+1)))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Time(s)")
    plt.ylabel("Control signal(m/s^2)")
    plt.legend()
    plt.show()


def distance_plotter(dt_list, distance_list, xref_list,split_car):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']

    for i in range(NUM_CARS - 1):
        plt.plot(dt_list, distance_list[i::NUM_CARS - 1], colors[i%7], label=('Δ' + str(i+1) + str(i+2)))
    plt.plot(dt_list, xref_list, '--k', label=("REF Δ" + str(split_car) + str(split_car+1)))
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel("Time(s)")
    plt.ylabel("Distance between vehciles(m)")
    plt.legend()
    plt.show()


def position_plotter(dt_list, x_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']

    for i in range(NUM_CARS):
        plt.plot(dt_list, x_list[i::NUM_CARS], colors[i%7], label=('x' + str(i+1)))
    plt.grid(True)
    plt.xlabel("Time(s)")
    plt.ylabel("Position(m)")
    plt.legend()
    plt.show()


def old_values_lists():
    ''' Define lists to hold old values for plotting the graphs '''
    x_list = []     # Vehicle positions
    v_list = []     # Vehicle velocities
    u_list = []     # Vehicle accelerations

    for i in range(NUM_CARS):
        ' The leading vehicle is the first index in the lists '
        x_list.append(X_LIST[i])
        v_list.append(V_INIT)
        u_list.append(U_INIT)

    # Distance between vehicles
    distance_list = []
    for i in range(NUM_CARS - 1):
        distance_list.append(x_list[i] - x_list[i+1] - LENGTH)

    dt_list = [0.0]            # Time list
    xref_list = [INIT_DIST]  # The reference for the splitting vehicle 
    return x_list, v_list, u_list, distance_list, dt_list, xref_list


def renew_acc(u_list, try_split, next_control_signals):
    u_list.append(0)    # Leader acc. 0 -> vel. const
    for i in range(NUM_CARS-1):
        try:
            u_list.append(next_control_signals[i])
        except:
            u_list.append(0)    # Let the control signal to be 0 if the mpc cannot solve
            try_split = False
    return u_list, try_split  


def renew_x_and_v(veh_states, x_list, v_list, distance_list):
    # Appends new states in position, velocity and acceleration list
    for i in range(NUM_CARS):
        v_list.append(veh_states[i].v)
        x_list.append(veh_states[i].x)

    for i in range(NUM_CARS - 1):
        distance_list.append(veh_states[i].x - veh_states[i+1].x - LENGTH)    
    
    return x_list, v_list, distance_list


def plot_graphs(v_list, distance_list, u_list, x_list, dt_list, xref_list, split_car):
    # Creating plots for velocity, distance, acceleration and position
    velocity_plotter(dt_list, v_list)
    distance_plotter(dt_list, distance_list, xref_list, split_car)
    acceleration_plotter(dt_list, u_list)
    position_plotter(dt_list, x_list)


def check_split(time, split_ready, try_split, hard_split, split_distance, xref, split_car):
    # Check if the prediction horizon can see the split position
    if (time + DT*PREDICTION_HORIZON) >= split_ready and try_split == True:
        last_ref_in_mpc = (time + PREDICTION_HORIZON * DT - split_ready)//DT     # How many of the last prediction horizon should have the hard constrain on the current distance
        if time >= split_ready - 3*DT: # If time is bigger than split time with 1 prediction horizon
            hard_split = False  # Cancel the hard split condition
        if time >= split_ready - 3*DT: # Change the reference so that the hard constraint becomes soft constraint, 2 time steps before split position
            xref[split_car-1] = split_distance + LENGTH     # Fix the reference for the vehicle we want to split. Instead of having it as hard constraint have it like soft constraint
    else:
        xref[split_car-1] = INIT_DIST + LENGTH # Make so that the reference for the splitting vehicle is initial if the split can not be accomplished and we haven't yet seen the split position 
        last_ref_in_mpc = -1     # None of the predictione horizon should have the distance as hard constrain
    return xref, int(last_ref_in_mpc), hard_split


def animation(inital_veh_states, split_event, initial_xref):
    '''Does the required calculations and plots the animation.'''

    veh_states = inital_veh_states    # Defining initial states as the current states
    xref = initial_xref               # Defining initial state reference as current reference

    # Longitudial pos, velocity, acceleration and time lists. These hold the neccesary information for plotting
    x_list, v_list, u_list, distance_list, dt_list, xref_list = old_values_lists()

    # Splitting conditions
    split_car = int(split_event[0])
    split_distance = int(split_event[1])
    split_ready = float(split_event[2])
    drag_or_not = split_event[3]

    time = 0.0
    try_split = True    
    hard_split = True   # The hard constraint is on
    x_last = None
    printer = True
    while MAX_TIME >= time:
        xref, last_ref_in_mpc, hard_split = check_split(time, split_ready, try_split, hard_split, split_distance, xref, split_car)  # Check if the split should start. If so put a hard constraint and change the ref near to the split position
        next_control_signals, x_last = mpc(veh_states, xref, split_car, last_ref_in_mpc, split_distance, drag_or_not, hard_split, x_last) # Calculate the next control signal
        u_list, try_split = renew_acc(u_list, try_split, next_control_signals)  # Renew the acceleration list 
        clear_and_draw_car(veh_states)                                          # Draw the cars on the simulation
        veh_states = update_states(veh_states, u_list, drag_or_not)          # Renew the vehicle states a for MPC
        x_list, v_list, distance_list = renew_x_and_v(veh_states, x_list, v_list, distance_list)
        xref_list.append(xref[split_car-1]-LENGTH)

        if try_split == False and ((veh_states[split_car-1].x - veh_states[split_car].x - LENGTH) > split_distance-1) and time <= split_ready - DT:
            try_split = True # If the split has been accomplished in that second even though the mpc could not calculate do not get back to initial position
        if try_split == False and ((veh_states[split_car-1].x - veh_states[split_car].x) < split_distance-1) and time >= split_ready and printer:
            print('The split could not been accomplished. Thus the split cancelled.')
            printer = False


        # Updates time
        time += DT
        time = round(time,1) # Albin --> la till den så att vi inte får 0.40000002. Istället så får vi nu 0.4 exakt
        dt_list.append(time)

        # Breakes animation when first vehicle reaches goal
        if veh_states[0].x >= ROAD_LENGTH:
            break

    plot_graphs(v_list, distance_list, u_list, x_list, dt_list, xref_list, split_car)


def pos_list_create():
    global NUM_CARS 
    global X_LIST

    NUM_CARS = int(input("Number of cars: "))
    X_POS = 0. # Initial position for the lead vehicle in the platoon is assumed to be at 0.0 meter
    
    "Albin. Jag har gjort det här för att annars så beror det på hur många bilar det finns i systemet."
    X_LIST = [X_POS] 
    for i in range(NUM_CARS-1):
        X_POS -= (INIT_DIST + LENGTH)  # LENGTH needs to be added because X_POS is the distance between the center of each vehicle while the distance is between the front-end and the back-end of the vehicle
        X_LIST.append(X_POS) 
    return


def split_event_finder():
    # Taking input on where, how and when the split should occur
    split_event = []    
    split_event.append(input("Split behind vehicle: "))
    split_event.append(input("Split distance: "))
    split_event.append(input("Split end position give in time: "))
    split_event.append(input('To simulate with air drag press 1 else 0: '))
    while split_event[-1] != '0' and split_event[-1] != '1':
        split_event[-1] = input('To simulate with air drag press 1 else 0: ')
    return split_event


def main():
    pos_list_create()   # Create the initial position list

    initial_veh_states = [] # Initial states for the vehicles. It consists of the position and the velocity of each vehicle
    initial_xref = [0]*(2*NUM_CARS-1)  # The reference state vector at the beginning

    for i in range(NUM_CARS):
        initial_veh_states.append(VehicleState(X_LIST[i], V_INIT)) # The initial vehicle states are the position at the moment and the initial velocity.
        initial_xref[NUM_CARS-1+i] = V_INIT    # The reference velocity

    for i in range(NUM_CARS-1):
        initial_xref[i] = INIT_DIST + LENGTH # The reference distance between the vehicles at the beginning

    split_event = split_event_finder()
    animation(initial_veh_states, split_event, initial_xref)


main()
