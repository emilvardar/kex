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
ROAD_LENGTH = 500 # [m]

# Start positions
NUM_CARS = 2  # NUMBER OF CARS
MAX_CARS = 7
X_START1 = 60.0
X_START2 = 50.0
X_START3 = 40.0
X_START4 = 30.0
X_START5 = 20.0
X_START6 = 20.0
X_START7 = 0.0
X_LIST = [X_START1, X_START2, X_START3, X_START4, X_START5, X_START6, X_START7]

# Initial acceleration
U_INIT = 0.0 # m/s^2

# Initial velocity
V_INIT = 80.0 / 3.6 # 80km/h -> 80/3.6 m/s

# Prediction horizon
PREDICTION_HORIZON = 500

# Split and non-split conditions
S_NON_SPLIT = X_START6 - X_START7
S_SPLIT = 2*S_NON_SPLIT

# Constriants
SAFETY_DISTANCE = 1.
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
        veh_states[i].v = veh_states[i].v + control_signals[-NUM_CARS + i] * DT
    for i in range(NUM_CARS-1):
        states[i].deltax =  veh_states[i].x -  veh_states[i+1].x
        states[i].deltav =  veh_states[i].v -  veh_states[i+1].v
    return veh_states, states


def MPC(states):
    # returns list with control signals for all vehicles
    us = optimization(states)
    return us


def optimization(states):
    """
    heavily inspired by https://www.cvxpy.org/tutorial/intro/index.html
    """
    Ad, Bd = create_matrices()
    #umin, umax, xmin, xmax = create_constraints()
    
    # Cost matrices
    R = 1.0 * sparse.eye(1)
    Q = sparse.diags([1,1])
    QN = Q

    # Define a for loop here for mult. vehicles
    x0 = np.array([states[0].deltax - 20, states[0].deltav]) # initial state
    #print(states[0].deltax)
    #print('hej')
    xr = np.array([0.0, 0.0]) #Dont know if neccesary (reference state)
    
    # Create two scalar optimization variables.
    u = cp.Variable((NUM_CARS-1, PREDICTION_HORIZON))
    x = cp.Variable((2*(NUM_CARS-1), PREDICTION_HORIZON+1))
    constraints = [x[:,0] == x0]
    cost = 0.0

    for k in range(PREDICTION_HORIZON):
        cost += cp.quad_form(xr - x[:,k], Q) + cp.quad_form(u[:,k], R)
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
        #constraints += [xmin <= x[:,k], x[:,k] <= xmax] # doesnt work for some reason
        constraints += [MIN_ACCELERATION <= u[:,k], u[:,k] <= MAX_ACCELERATION]
    cost += cp.quad_form(x[:,PREDICTION_HORIZON] - xr, QN)
    
    # Form and solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    sol = prob.solve(solver=cp.SCS)
    #print(sol)
    #print(u.value[:,0])
    return u.value[:,0]


def create_constraints():
    umin = np.array([MIN_ACCELERATION, MIN_ACCELERATION])
    umax = np.array([MAX_ACCELERATION, MAX_ACCELERATION])
    xmin = np.array([SAFETY_DISTANCE, -np.inf])
    xmax = np.array([np.inf, np.inf])
    return umin, umax, xmin, xmax


def create_matrices():
    A = ([
        [0.,1.],
        [0.,0.]
    ])
    B = ([
        [0., -1.]
    ])
    return A, B


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

    plt.axis([0, 500, -50, 50])
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
        distance_list.append(x_list[i] - x_list[i + 1])

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


def animation(inital_veh_states, initial_states):
    '''Does the required calculations and plots the animation.'''

    veh_states = inital_veh_states
    states = initial_states

    # Longitudial pos, velocity and acceleration lists. These hold the neccesary information for plotting
    x_list, v_list, u_list, distance_list, dt_list = old_values_lists()
    time = 0.0

    while 5*MAX_TIME >= time:
        us = MPC(states)
        u_list.append(0)
        u_list.append(us[0])
        
        clear_and_draw_car(veh_states)
        veh_states, states = update_states(veh_states, states, u_list)
        
        # appends new states in position, velocity and acceleration list
        for i in range(NUM_CARS):
            v_list.append(veh_states[i].v)
            x_list.append(veh_states[i].x)

        for i in range(NUM_CARS - 1):
            distance_list.append(veh_states[i].x - veh_states[i+1].x)

        # updates time
        time += DT
        dt_list.append(time)

        # breakes animation when first vehicle reaches goal
        if veh_states[0].x >= ROAD_LENGTH:
            break

    # plots
    
    velocity_plotter(dt_list, v_list)
    distance_plotter(dt_list, distance_list)
    acceleration_plotter(dt_list, u_list)
    position_plotter(dt_list, x_list)
    

def main():
    initial_states = []
    initial_veh_states = []
    for i in range(NUM_CARS):
        initial_veh_states.append(VehicleState(X_LIST[MAX_CARS - NUM_CARS + i], V_INIT))

    for i in range(NUM_CARS-1):
        initial_states.append(State(initial_veh_states[i], initial_veh_states[i+1]))

    # initial_states will look like = [leader car, car2, car3, car4 ....] -> car is an object an looks like car.x = X, car.v = V
    animation(initial_veh_states, initial_states)

main()
