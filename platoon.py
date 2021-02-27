import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cp

# Vehicle parameters
PARA = 1 #Use to minimize the vehicle parameters
LENGTH = 4.5/PARA  # [m]
WIDTH = 2.0/PARA  # [m]
BACKTOWHEEL = 1.0/PARA  # [m]
WHEEL_LEN = 0.3/PARA  # [m]
WHEEL_WIDTH = 0.2/PARA  # [m]
TREAD = 0.7/PARA  # [m]
WB = 2.5/PARA  # [m]

# Road length
ROAD_LENGTH = 300

# Start positions
NUM_CARS = 2 #NUMBER OF CARS
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
U_INIT = 0.0

# Initial velocity
V_INIT = 60.0

# Prediction horizon
PREDICTION_HORIZON = 5

# Constriants
SAFETY_DISTANCE = 1
MAX_VELOCITY = 120/3.6 # 120km/h -> 120/3.6 m/s
MIN_VELOCITY = 60/3.6 # 60km/h -> 60/3.6 m/s
MAX_ACCELERATION = 0.5 # 0.5 m/s^2

# Cost matrices
R = np.diag([0.01, 0.01])
Q = np.diag([0.01, 0.01])

# Max time
MAX_TIME = 30 #[s]
# Time step
DT = 0.5 # [s]

class VehicleState:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, v=0.0):
        self.x = x
        self.v = v

def update_states(states,control_signals):
    # updates states for all vehicles
    for i in range(NUM_CARS):

        states[i].x = states[i].x + states[i].v * DT
        states[i].v = states[i].v + control_signals[i] * DT

    return states

def MPC(states):
    # returns list with control signals for all vehicles
    control_signals = optimization(states)
    return control_signals

def optimization(states):
    """
    heavily inspired by https://www.cvxpy.org/tutorial/intro/index.html
    """

    # Create two scalar optimization variables.
    x = cp.Variable(( 2*(NUM_CARS-1), PREDICTION_HORIZON ))
    u = cp.Variable(( NUM_CARS, PREDICTION_HORIZON ))

    # Create two constraints.
    constraints = []
    cost = 0.0
    A, B = create_matrices()
    for t in range(PREDICTION_HORIZON):
            
        cost += cp.quad_form(u[:, t], R)

        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]
 
        cost += cp.quad_form(x[:, t], Q)

    for i in range(NUM_CARS-1):

        constraints += [x[2*i,:] - x[2*(i+1),:] >= SAFETY_DISTANCE]     #Minimum distance between vehicles
        constraints += [x[(2*i)+1,:] <= MAX_VELOCITY]   # Maximum velocity 
        constraints += [x[(2*i)+1,:] >= MIN_VELOCITY]   #Minimum velocity

        constraints += [u[i+1,:] <= MAX_ACCELERATION]
        constraints += [u[i+1,:] >= -MAX_ACCELERATION]

        #constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

    # Form objective.
    obj = cp.Minimize(cost)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value, u.value)

    return u.value[:,0]

def create_constraints(states, x, u, A, B):
    

    return constraints

def create_matrices():

    A = [[0,1],
        [0,0]]
    
    B = [[0,0],
        [1,-1]]

    return A,B

def plot_car(x, y=0.0, yaw=0.0, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

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

def distance(distance_list, temp_disp_list):
    '''Calculates the distance between the ancestor car'''
    new_distance_list = []
    for i in range(NUM_CARS - 1):
        delta = temp_disp_list[-NUM_CARS] - temp_disp_list[-NUM_CARS + i]
        new_distance_list.append(distance_list[-NUM_CARS + 1 + i] + delta)
    for m in range(len(new_distance_list)):
        distance_list.append(new_distance_list[m])
    return distance_list

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
        plot_car(states[i].x) # plot the cars

    plt.axis([0, 300, -50, 50])
    plt.grid(True)
    plt.pause(0.0001)

def velocity_plotter(dt_list, v_list):
    plt.subplots(1)
    colors = ['-b', '-g', '-r', '-c', '-m', '-y', '-k']
    labels = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    for i in range(NUM_CARS):
        plt.plot(dt_list, v_list[i::NUM_CARS], colors[i], label = labels[i])

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
        plt.plot(dt_list, u_list[i::NUM_CARS], colors[i], label = labels[i])
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
    for i in range(NUM_CARS-1):
        plt.plot(dt_list, distance_list[i::NUM_CARS-1], colors[i], label = labels[i])
    plt.grid(True)
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.legend()
    plt.show()

def old_values_lists():
    ''' Define list to hold old values for plotting the graphs '''

    # Initial car positions
    x_list = []
    for i in range(NUM_CARS):
        x_list.append(X_LIST[MAX_CARS-NUM_CARS+i])

    # Initial velocities
    v_list = []
    for i in range(NUM_CARS):
        v_list.append(V_INIT)

    # Initial accelerations
    u_list = []
    for i in range(NUM_CARS):
        u_list.append(U_INIT)

    # Distance between the cars at the begining
    distance_list = []
    for i in range(NUM_CARS-1):
        distance_list.append(x_list[i] - x_list[i+1])

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

def animation(inital_states):
    '''Does the required calculations and plots the animation.'''

    states = inital_states

    # Coordinate, velocity and acceleration lists. These are for plotting.
    x_list, v_list, u_list, distance_list, dt_list = old_values_lists()

    time = 0.0

    while MAX_TIME >= time:
        clear_and_draw_car(states)

        control_signals = MPC(states)

        states  = update_states(states,control_signals)

        #appends new states in position, velocity and acceleration list
        for i in range(NUM_CARS):
            v_list.append(states[i].v)
            x_list.append(states[i].x)
            u_list.append(control_signals[i])

        #The displacement in 1 iteration is calculated by x = v*dt
        # And the new position is calculated by the old position + the displacement
        temp_disp_list = []
        for i in range(NUM_CARS):
            displacement = v_list[-NUM_CARS + i] * DT
            temp_disp_list.append(displacement)

        distance_list = distance(distance_list, temp_disp_list)

        #updates time
        time += DT
        dt_list.append(time)

        #breakes animation when first vehicle reaches goal
        if states[0].x >= ROAD_LENGTH:
            break
    
    # plots
    velocity_plotter(dt_list, v_list)
    distance_plotter(dt_list, distance_list)
    acceleration_plotter(dt_list, u_list)
    position_plotter(dt_list, x_list)

def main():
    initial_states = []
    for i in range(NUM_CARS):
        initial_states.append(VehicleState(X_LIST[MAX_CARS - NUM_CARS + i], 60.0))
    
    animation(initial_states)

main()
