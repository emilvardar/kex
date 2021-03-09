import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cp
import scipy as sp
from scipy import sparse

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
ROAD_LENGTH = 300 # [m]

# Start positions
NUM_CARS = 5  # NUMBER OF CARS
MAX_CARS = 7
X_START1 = 60.0
X_START2 = 50.0
X_START3 = 40.0
X_START4 = 30.0
X_START5 = 20.0
X_START6 = 10.0
X_START7 = 0.0
X_LIST = [X_START1, X_START2, X_START3, X_START4, X_START5, X_START6, X_START7]

# Initial acceleration
U_INIT = 0.0 # m/s^2

# Initial velocity
V_INIT = 80.0 / 3.6 # 80km/h -> 80/3.6 m/s

# Prediction horizon
PREDICTION_HORIZON = 30

# Split and non-split conditions
S_NON_SPLIT = X_START6 - X_START7
S_SPLIT = 2*S_NON_SPLIT

# Constriants
SAFETY_DISTANCE = 1. + LENGTH
MAX_VELOCITY = 120 / 3.6  # 120km/h -> 120/3.6 m/s
MIN_VELOCITY = 60 / 3.6  # 60km/h -> 60/3.6 m/s
MAX_ACCELERATION = 5  # 5 m/s^2
MIN_ACCELERATION = -5 # 10 m/s^2 decleration
MAX_VEL_DIFF = 20 # m/s
MIN_VEL_DIFF = -20 # m/s

# Max time
MAX_TIME = 5  # [s]
# Time step
DT = 0.1  # [s]


class VehicleState:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, v=0.0):
        self.x = x
        self.v = v


class State:
    def __init__(self, vehicle_front, vehicle_back):
        self.deltax = vehicle_front.x - vehicle_back.x
        self.deltav = vehicle_front.v - vehicle_back.v


def update_states(veh_states, states, control_signals):
    # updates states for all vehicles
    for i in range(NUM_CARS):
        veh_states[i].x = veh_states[i].x + veh_states[i].v * DT
        veh_states[i].v = veh_states[i].v + control_signals[-NUM_CARS + i]* DT
        """ adding air-drag
        if i != 0:
            veh_states[i].v = veh_states[i].v + (control_signals[-NUM_CARS + i] - 0.001*(veh_states[i].v)**2) * DT
        else:
            veh_states[i].v = veh_states[i].v + control_signals[-NUM_CARS + i]* DT
        """
    for i in range(NUM_CARS-1):
        states[i].deltax =  veh_states[i].x - veh_states[i+1].x
        states[i].deltav =  veh_states[i].v - veh_states[i+1].v
    return veh_states, states


def MPC(states, xref):
    """
    heavily inspired by https://www.cvxpy.org/tutorial/intro/index.html
    """
    Ad, Bd = create_matrices()
    #umin, umax, xmin, xmax = create_constraints()
    
    # Cost matrices
    R = 1.0 * sparse.eye(NUM_CARS-1)
    
    '''Mjukkodad Q och xref'''
    q_vec = []
    #xref = []
    # Kan även ha if satser i for loopen om man skulle vilja ha olika xref och Q beroende på vilken bil man vill splitta.
    for i in range(NUM_CARS-1):
        q_vec.append(10.)
        q_vec.append(0.1)
        #xref.append(20.+LENGTH)
        #xref.append(0.)
    Q = sparse.diags(q_vec)
    QN = Q
    '''
    Hardkodad Q och xref
    if NUM_CARS == 2:
        Q = sparse.diags([10.0, 0.1])
        xref = [(20.0 + LENGTH), 0.0]
    elif NUM_CARS == 3:
        Q = sparse.diags([10.0, 0.1, 10.0, 0.1])
        xref = [(-10.0 + LENGTH), 0.0, (-10.0 + LENGTH), 0.0]
    elif NUM_CARS == 4:
        Q = sparse.diags([10.0, 0.1, 10.0, 0.1, 10., 0.1])
        xref = [(5.0 + LENGTH), 0.0, (20.0 + LENGTH), 0.0, (5.0 + LENGTH), 0.0]
    QN = Q
    '''

    # Define a for loop here for mult. vehicles
    #x0 = np.array([states[0].deltax, states[0].deltav]) # initial state
    x0 = []
    for i in range(NUM_CARS-1):
        x0.append(states[i].deltax)
        x0.append(states[i].deltav)
    
    # Create two scalar optimization variables.
    u = cp.Variable((NUM_CARS-1, PREDICTION_HORIZON))
    x = cp.Variable((2*(NUM_CARS-1), PREDICTION_HORIZON+1))
    # First constraints needs to be the initial state
    constraints = [x[:,0] == x0]
    # Define the cost function
    cost = 0.0

    for k in range(PREDICTION_HORIZON):
        cost += cp.quad_form(x[:,k] - xref, Q) + cp.quad_form(u[:,k], R)    # Add the cost function sum(x^TQx + u^TRu) 
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]                  # Add the system x(k+1) = Ax(k) + Bu(k)
        for i in range(NUM_CARS-1):                                         # The for loop is for defining safety distance for all cars
            constraints += [SAFETY_DISTANCE <= x[2*i,k]]
        constraints += [[MIN_ACCELERATION] <= u[:,k], u[:,k] <= [MAX_ACCELERATION]] # Constarints for the acc.
    cost += cp.quad_form(x[:,PREDICTION_HORIZON] - xref, QN)
    
    # Form and solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    sol = prob.solve(solver=cp.ECOS)
    #print(u.value)
    return u[:,0].value


def create_constraints():
    umin = np.array([MIN_ACCELERATION, MIN_ACCELERATION])
    umax = np.array([MAX_ACCELERATION, MAX_ACCELERATION])
    xmin = np.array([SAFETY_DISTANCE, -np.inf])
    xmax = np.array([np.inf, np.inf])
    return umin, umax, xmin, xmax


def create_matrices():

    # Define dimentions for A and B matrices
    A = np.zeros((2*(NUM_CARS-1),2*(NUM_CARS-1)))
    B = np.zeros((2*(NUM_CARS-1),(NUM_CARS-1)))

    j = 0
    for i in range(2*(NUM_CARS-1)):

        if i in range(0,2*(NUM_CARS-1),2): # Only odd rows have non-zero elements in them
            A[i,i+1] = 1    # A-matrix

            B[i+1,j] = -1   # B-matrix
            if j != 0:
                B[i+1,j-1] = 1
            j += 1

    # Identity matrix
    I = sparse.eye(2*(NUM_CARS-1))

    # Discretize A and B matrices
    Ad = (I + DT*A)
    Bd =  DT*B
    return Ad, Bd


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
    labels = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    for i in range(NUM_CARS):
        plt.plot(dt_list, v_list[i::NUM_CARS], colors[i], label=labels[i])

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("time(s)")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()


def acceleration_plotter(dt_list, u_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']
    labels = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    for i in range(NUM_CARS):
        plt.plot(dt_list, u_list[i::NUM_CARS], colors[i], label=labels[i])
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("time(s)")
    plt.ylabel("acceleration(m/s^2)")
    plt.legend()
    plt.show()


def distance_plotter(dt_list, distance_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']
    labels = ['Δ12', 'Δ23', 'Δ34', 'Δ45', 'Δ56', 'Δ67']
    for i in range(NUM_CARS - 1):
        plt.plot(dt_list, distance_list[i::NUM_CARS - 1], colors[i], label=labels[i])
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.legend()
    plt.show()


def old_values_lists():
    ''' Define list to hold old values for plotting the graphs '''

    # Initial car positions
    x_list = []
    # Initial velocities
    v_list = []
    # Initial accelerations
    u_list = []
    for i in range(NUM_CARS):
        x_list.append(X_LIST[MAX_CARS - NUM_CARS + i])
        v_list.append(V_INIT)
        u_list.append(U_INIT)

    # Distance between the cars at the begining
    distance_list = []
    for i in range(NUM_CARS - 1):
        distance_list.append(x_list[i] - x_list[i + 1] - LENGTH)

    # Time list
    dt_list = [0]

    return x_list, v_list, u_list, distance_list, dt_list


def position_plotter(dt_list, x_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']
    labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']

    for i in range(NUM_CARS):
        plt.plot(dt_list, x_list[i::NUM_CARS], colors[i], label=labels[i])
    plt.grid(True)
    plt.xlabel("time(s)")
    plt.ylabel("position(m)")
    plt.legend()
    plt.show()


def animation(inital_veh_states, initial_states, split_event, initial_xref):
    '''Does the required calculations and plots the animation.'''

    # Defining initial states as the current states
    veh_states = inital_veh_states
    states = initial_states
    xref = initial_xref

    # Longitudial pos, velocity and acceleration lists. These hold the neccesary information for plotting
    x_list, v_list, u_list, distance_list, dt_list = old_values_lists()

    # Spliting condition
    split_car = int(split_event[0])
    split_distance = int(split_event[1])
    split_start_position = float(split_event[2])
    split_end_position = float(split_event[3])
    #print(initial_xref[2*(split_car-1)])
    # The simulation is done in this while-loop
    time = 0.0
    while 5*MAX_TIME >= time:
        
        # Changes the positional reference if the spliting vehicle is between "start" and "end" position
        if veh_states[split_car-1].x >= split_start_position*0.9 and veh_states[split_car-1].x < split_end_position*0.9:
            xref[2*(split_car-1)] = split_distance + LENGTH
        # Returns the positional reference back to normal when "end" position is reached
        if veh_states[split_car-1].x >= split_end_position*0.9:
            xref[2*(split_car-1)] = 10. +LENGTH#initial_xref[2*(split_car-1)]

        next_control_signals = MPC(states, xref)
        u_list.append(0)    # Leader acc. 0 -> vel. const
        #u_list.append(time/2)  #leader acc. const
        #u_list.append(math.sin(time))  #leader acc. sin

        for i in range(NUM_CARS-1):
            u_list.append(next_control_signals[i])
        
        clear_and_draw_car(veh_states)
        veh_states, states = update_states(veh_states, states, u_list)
        
        # Appends new states in position, velocity and acceleration list
        for i in range(NUM_CARS):
            v_list.append(veh_states[i].v)
            x_list.append(veh_states[i].x)

        for i in range(NUM_CARS - 1):
            distance_list.append(veh_states[i].x - veh_states[i+1].x - LENGTH)

        # Updates time
        time += DT
        dt_list.append(time)

        # Breakes animation when first vehicle reaches goal
        if veh_states[0].x >= ROAD_LENGTH:
            break

    # Creating plots for velocity, distance, acceleration and position
    velocity_plotter(dt_list, v_list)
    distance_plotter(dt_list, distance_list)
    acceleration_plotter(dt_list, u_list)
    position_plotter(dt_list, x_list)
    

def main():
    initial_states = []
    initial_veh_states = []
    initial_xref = []

    for i in range(NUM_CARS):
        initial_veh_states.append(VehicleState(X_LIST[MAX_CARS - NUM_CARS + i], V_INIT))
        
    for i in range(NUM_CARS-1):
        initial_states.append(State(initial_veh_states[i], initial_veh_states[i+1]))

        initial_xref.append(initial_veh_states[i].x - initial_veh_states[i+1].x)
        initial_xref.append(0.)

    # initial_states will look like = [car1(Leader), car2, car3, car4 ....] -> car is an object an looks like car.x = X, car.v = V

    # Taking input on where, how and when the split should occur
    split_event = []
    split_event.append(input("Split behind vehicle: "))
    split_event.append(input("Split distance: "))
    split_event.append(input("Split start position: "))
    split_event.append(input("Split end position: "))

    animation(initial_veh_states, initial_states, split_event, initial_xref)

main()
