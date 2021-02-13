import matplotlib.pyplot as plt
import numpy as np
import math
from KEX_important import cubic_spline_planner

# Vehicle parameters
PARA = 1 #Use to minimize the vehicle parameters
LENGTH = 4.5/PARA  # [m]
WIDTH = 2.0/PARA  # [m]
BACKTOWHEEL = 1.0/PARA  # [m]
WHEEL_LEN = 0.3/PARA  # [m]
WHEEL_WIDTH = 0.2/PARA  # [m]
TREAD = 0.7/PARA  # [m]
WB = 2.5/PARA  # [m]

# start positions
X_START1 = 20
X_START2 = 10
X_START3 = 0

# Constrains on accelaration
U_MIN = -1.5
U_MAX = 1.5
U_INIT = 0

# Constrains on velocity
V_MAX = 120
V_MIN = 0
V_INIT = 60

#Constrains on distance
S0 = 5 # 5 meter between the cars is the minimum possible.

# time step
DT = 0.1/6

#Parameters for cost function
C1 = 0.1
C2 = 1
C3 = 0.5

# initialy distance between the cars
S_REF = X_START1 - X_START2
# reference distance when vehicles should split
S_REF_SPLIT = 2 * S_REF


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

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

def get_straight_course(dl):
    a = 5 # A parameter to make the trajectory longer
    ax = [a*0.0, a*5.0, a*10.0, a*20.0, a*30.0, a*40.0, a*50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def distance(prev_distance, disp1, disp2):
    '''Calculates the distance between the ancestor car'''
    delta = disp1 - disp2
    new_distance = prev_distance + delta
    return new_distance


def initials():
    '''Initial values for position, velocity and accelaration'''
    # Initial car positions
    s1 = X_START1
    s2 = X_START2
    s3 = X_START3

    # Initial speeds
    v1 = V_INIT
    v2 = V_INIT
    v3 = V_INIT

    # Initial accelarations
    u1 = U_INIT
    u2 = U_INIT
    u3 = U_INIT

    return s1, s2, s3, v1, v2, v3, u1, u2, u3


def clear_and_draw_car(x1, x2, x3):
    plt.cla()
    dl = 1  # course tick
    cx, cy, cyaw, ck = get_straight_course(dl)  #get the straight line
    plt.plot(cx, cy, "-r", label="course")      #plot the cars
    plot_car(x1, 0, 0, steer=0.0)
    plot_car(x2, 0, 0, steer=0.0)
    plot_car(x3, 0, 0, steer=0.0)
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)


def velocity_plotter(dt_list, v1_list, v2_list, v3_list):
    plt.subplots(1)
    plt.plot(dt_list, v1_list, "-b", label="v1")
    plt.plot(dt_list, v2_list, "-g", label="v2")
    plt.plot(dt_list, v3_list, "-r", label="v3")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("time(s)")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()


def distance_plotter(dt_list, distance12_list, distance23_list):
    plt.subplots(1)
    plt.plot(dt_list, distance12_list, "-b", label="Distance between car 1 and 2")
    plt.plot(dt_list, distance23_list, "-r", label="Distance between car 2 and 3")
    plt.grid(True)
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.legend()
    plt.show()


def animation():
    '''Does the required calculations and plots the animation.'''

    # Define list to hold old values for plotting the graphs
    # Velocity lists
    v1_list = [V_INIT]
    v2_list = [V_INIT]
    v3_list = [V_INIT]

    # Accelaration lists
    u1_list = [U_INIT]
    u2_list = [U_INIT]
    u3_list = [U_INIT]

    # Time list
    dt_list = [0]

    # Initialize the screen and the cars
    s1, s2, s3, v1, v2, v3, u1, u2, u3 = initials()

    # Distance between the cars at the begining
    distance_12 = s1 - s2
    distance_23 = s2 - s3

    # Position lists
    distance12_list = [distance_12]
    distance23_list = [distance_23]

    for i in range(200):

        clear_and_draw_car(s1, s2, s3)

        # If the accelartion is different from 0 the velocity for the car should change according to a = dv/dt -> dv = a*dt
        # And the total velocitiy is then v_new = v_old + dt
        v1 = u1 * DT + v1
        v1 = np.clip(v1, V_MIN, V_MAX)
        v1_list.append(v1)
        v2 = u2 * DT + v2
        v2 = np.clip(v2, V_MIN, V_MAX)
        v2_list.append(v2)
        v3 = u3 * DT + v3
        v3 = np.clip(v3, V_MIN, V_MAX)
        v3_list.append(v3)
        dt_list.append(DT+dt_list[-1])

        #The displacement in 1 iteration is calculated by x = v*dt
        # And the new position is calculated by the old position + the displacement
        displacement1 = v1 * DT
        s1 = s1 + displacement1
        displacement2 = v2 * DT
        s2 = s2 + displacement2
        displacement3 = v3 * DT
        s3 = s3 + displacement3

        distance_12 = distance(distance_12, displacement1, displacement2)
        distance12_list.append(distance_12)
        distance_23 = distance(distance_23, displacement2, displacement3)
        distance23_list.append(distance_23)

    velocity_plotter(dt_list, v1_list, v2_list, v3_list)            #Same logic can be used to plot the distance between the cars,
                                                                    #position of the cars and accelaration of each car.
    distance_plotter(dt_list, distance12_list, distance23_list)
def main():
    animation()

main()


