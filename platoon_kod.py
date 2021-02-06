# import turtle package
import turtle
import numpy as np

U_MIN = -1.5
U_MAX = 1.5
U_INIT = 0

V_MAX = 100
V_INIT = 60

X_START1 = -250
X_START2 = -300
X_START3 = -350

DT = 0.005

#Parameters for cost function
C1 = 0.1
C2 = 1
C3 = 0.5


def build_car(car):

    car.fillcolor('black')   #Make the car black
    car.begin_fill()          #Start color filling
    car.circle(10)           #Make car as a circle
    car.end_fill()           #End color filling


def create_screen():

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
    # create a turtle object
    car1 = turtle.Turtle()
    car2 = turtle.Turtle()
    car3 = turtle.Turtle()

    # hide turtle object (it hides the arrow)
    car1.hideturtle()
    car2.hideturtle()
    car3.hideturtle()

    # set initial position (50m between them)
    car1.goto(X_START1, 0)
    car2.goto(X_START2, 0)
    car3.goto(X_START3, 0)

    return car1, car2, car3

def distance(total_distance, disp1, disp2):
    change_dist = disp1 - disp2
    new_distance = total_distance + change_dist
    return new_distance



def MPC():  #Is not done yet
    l_s_prim = -2*C1*(s[i] - (v[i]*tg+s0))
    l_v_prim = 2*C1*(s[i] - (v[i]*tg+s0))*tg + 2*C2*(v[i-1]-v[i]) + l_s_prim
    u_star = -l_v_prim/(2*C3)
    u_star = np.clip(u_star, U_MIN, U_MAX)


def animation():
    # infinite loop
    screen = create_screen()
    car1, car2, car3 = movement_initializer()

    # Divide with 400 so we get a better animation
    v1 = V_INIT
    v2 = V_INIT
    v3 = V_INIT

    # Accelaration should be 0 at the beginning
    u1 = U_INIT
    u2 = U_INIT
    u3 = U_INIT

    distance_12 = X_START1 - X_START2
    distance_23 = X_START2 - X_START3

    i = 0
    while True:
        i = i + 1
        # clear turtle work
        car1.clear()
        car2.clear()
        car3.clear()

        # call function to draw ball
        build_car(car1)
        build_car(car2)
        build_car(car3)

        # update screen
        screen.update()

        #Find the new values for the velocities
        v1 = u1 * DT + v1
        v2 = u2 * DT + v2
        v3 = u3 * DT + v3

        #The displacement in 1 iteration is calculated by x = v*dt
        displacement1 = v1 * DT
        displacement2 = v2 * DT
        displacement3 = v3 * DT

        distance_12 = distance(distance_12, displacement1, displacement2)
        distance_23 = distance(distance_23, displacement2, displacement3)

        if i % 100 == 0:
            print(v3)
            print(distance_12)
            print('HEJ')
            print(distance_23)
            print('HEJ2')

        # forward motion by turtle object
        car1.forward(displacement1)
        car2.forward(displacement2)
        car3.forward(displacement3)

        if i > 5000:
            break

def main():
    animation()
main()


