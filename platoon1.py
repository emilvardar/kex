# import turtle package
import turtle
import matplotlib.pyplot as plt

U_MIN = -1.5
U_MAX = 1.5
U_INIT = 0

V_MAX = 100
V_INIT = 60

# start positions
X_START1 = -400
X_START2 = -450
X_START3 = -500

# initialy distance between the cars
TG = X_START1 - X_START2

# time step
DT = 0.1

#Parameters for cost function
C1 = 0.1
C2 = 1
C3 = 0.5

def build_car(car,color):
    '''Creates a car. The dots can be interpreted as simplified cars'''
    car.fillcolor(color)   #Make the car black
    car.begin_fill()          #Start color filling
    car.circle(10)           #Make car as a circle
    car.end_fill()           #End color filling


def create_screen():
    '''Creatas background for the simulation.'''
    # create a screen object
    screen = turtle.Screen()

    # set screen size
    screen.setup(1000, 400)

    # screen background color
    screen.bgcolor('white')

    # screen updaion
    screen.tracer(0)
    return screen

def movement_initializer():
    '''Creates the cars and set them to the initial position'''
    # create a turtle object
    car1 = turtle.Turtle()
    car2 = turtle.Turtle()
    car3 = turtle.Turtle()
    merge_point = turtle.Turtle()

    # hide turtle object (it hides the arrow)
    car1.hideturtle()
    car2.hideturtle()
    car3.hideturtle()
    merge_point.hideturtle()

    # set initial position
    car1.goto(X_START1, 0)
    car2.goto(X_START2, 0)
    car3.goto(X_START3, 0)
    merge_point.goto(0,0)

    return car1, car2, car3, merge_point

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


def clear_and_draw(car1, car2, car3, merge_point, screen):
    '''Clears old position of the car and plots the new position'''
    # clear turtle work
    car1.clear()
    car2.clear()
    car3.clear()
    merge_point.clear()

    # call function to draw ball
    build_car(car1,'black')
    build_car(car2,'black')
    build_car(car3,'black')
    build_car(merge_point,'red')

    # update screen
    screen.update()


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

def animation():
    '''Does the required calculations and plots the animation.'''

    v1_list = [V_INIT]
    v2_list = [V_INIT]
    v3_list = [V_INIT]
    dt_list = [0]

    # Initialize the screen and the cars
    screen = create_screen()
    car1, car2, car3, merge_point = movement_initializer()

    s1, s2, s3, v1, v2, v3, u1, u2, u3 = initials()

    # Distance between the cars at the begining
    distance_12 = X_START1 - X_START2
    distance_23 = X_START2 - X_START3

    for i in range(5000):

        clear_and_draw(car1, car2, car3, merge_point, screen)

        # If the accelartion is different from 0 the velocity for the car should change according to a = dv/dt -> dv = a*dt
        # And the total velocitiy is then v_new = v_old + dt
        v1 = u1 * DT + v1
        v1_list.append(v1)
        v2 = u2 * DT + v2
        v2_list.append(v2)
        v3 = u3 * DT + v3
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
        distance_23 = distance(distance_23, displacement2, displacement3)

        # forward motion by turtle object
        car1.forward(displacement1/20) #displacement is now v1*DT /approx = 60*0.005 = 0.3. If DT=0.1 than displacement /approx = 60*0.1 = 6.
        # So divede this with 20. This is just for the sake of animation
        car2.forward(displacement2/20)
        car3.forward(displacement3/20)

        #Some tests
        #if i % 100 == 0:
        #    print(v3)
        #    print(distance_12)
        #    print('HEJ')
        #    print(distance_23)
        #    print('HEJ2')

    velocity_plotter(dt_list, v1_list, v2_list, v3_list) #Same logic can be used to plot the distance between the cars,
    #position of the cars and accelaration of each car.

def main():
    animation()

main()


