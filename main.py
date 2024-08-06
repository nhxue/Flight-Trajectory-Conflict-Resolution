#Heading change algorithm based on paper by Ralph Bach and Chris Farrell: citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=76e15e2383474e7cc85bb433a5785632dab6a80d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import random



#calculates the time from first heading change to turnback (when relative velocity vector is left of position vector)
def turnback_time(distance, alpha, mu, vr):
    dtb = (distance * np.cos(alpha))/np.cos(mu)
    ttb = dtb/vr
    return (ttb, dtb)

#calculates the time from first heading change to turnback (when relative velocity vector is right of position vector)
def turnback_time2(distance, alpha, mu, vr):
    dtb = (distance * np.cos(alpha))/np.cos(mu+2*alpha)
    ttb = dtb/vr
    return (ttb, dtb)

#output range is [0, 2pi]; accounts for x values less than 0 (more robust arctan operator)
def arctan_new(y,x):
    theta = np.arctan(y/x)
    if x < 0: 
        theta = theta + np.pi
    return theta + np.pi*2


'''
the 'update' function plots the current coordinates of both aircraft (always updating) and 
also updates display metrics such as projected minimum separation, the current distance,
the dot product between the line of sight vector and relative velocity vector, and the minimum separation 
up to that point.
 '''
def update(frame):
    if conflict == False:
        print('No conflict detected')
        exit()

    #stops plotting if frames end
    if frame >= fps * total_time_s - 2:
        exit()

    #constantly updating position
    global min_distance, dmpc
    object1.center = (xa[frame], ya[frame])
    object2.center = (xb[frame], yb[frame])
    radius1.center = (xa[frame], ya[frame])
    radius2.center = (xb[frame], yb[frame])

    #calculate distance between centers of both aircraft
    distance = np.sqrt((xa[frame] - xb[frame])**2 + (ya[frame] - yb[frame])**2)

    #turns the two aircraft's markers red if current separation distance exceeds minimum allowable separation
    if distance <= radius1.get_radius():
        radius1.set_color('red')
        radius2.set_color('red')
    else:
        radius1.set_color('black')
        radius2.set_color('black')
    
    '''assigns aircraft a's heading based on where it is on its path: 
    t1 = original path before heading change, 
    t2 = path after initial heading change (to avoid collision),
    t3 = path as aircraft a turns back towards its original path,
    t4 = return to original path'''
    if frame in t1_frames:
        phia = theta1
    elif frame in t2_frames:
        phia = new_heading_angle
    elif frame in t3_frames: 
        phia = turnback_angle
    elif frame in t4_frames:
        phia = theta1

    '''continuously updating calculations while plotting 
    N = y-component of relative velocity vector
    D = x-component of relative velocity vector
    Vr = magnitude of relative velocity vector
    determining minimum separation distance using angle between relative velocity vector and LoS vector (line of sight)'''
    phib = theta2
    curr_N = v1 * np.sin(phia) - v2 * np.sin(phib)
    curr_D = v1 * np.cos(phia) - v2 * np.cos(phib)
    curr_Vr = np.sqrt(np.power(curr_N, 2) + np.power(curr_D, 2))
    
    curr_cosalpha = (curr_D * (xb[frame]-xa[frame]) + curr_N * (yb[frame]-ya[frame]))/((curr_Vr)*distance)
    curr_sinalpha = np.sqrt(1 - np.power(curr_cosalpha, 2))
    current_rms = distance * curr_sinalpha
    
    #plots rms (minimum separation) and cosalpha (indicator of whether there is a collision or not using dot product)
    rms_text.set_text(f'rms: {current_rms:.2f} nmi, cos(alpha) = {curr_cosalpha:.2f}')

    '''if predicted min separation is less than min allowable separation 
    and cosalpha > 0 (LoS and relative velocity vectors are in same direction)
    then turn text red to indicate predicted collision'''
    if current_rms + tolerance < min_separation_nmi and curr_cosalpha > 0:
        conflict_flag = True
        rms_text.set_color('r')
    else:
        conflict_flag = False
        rms_text.set_color('k')
    

    #plots locations and updates distance
    text.set_text(f'Distance: {distance:.2f} nmi')
    plt.plot(xa[frame], ya[frame], marker = '.', color='blue', markersize = 0.5)
    plt.plot(xb[frame], yb[frame], marker = '.', color='purple', markersize = 0.5)
    
    #updating minimum distance
    if distance < min_distance:
        min_distance = distance
        record_min_distance.set_text(f'Minimum Distance (so far): {min_distance:.2f} nmi')
    
    #displays time to first loss (time to collision without heading change)
    if (current_rms < min_separation_nmi):
        dfl = (distance * curr_cosalpha) - np.sqrt(min_separation_nmi**2-current_rms**2)
        tfl = dfl/curr_Vr
        if tfl >= 0:
            t_conflict.set_text(f'Time to first loss: {tfl:.2f} seconds')
    
    return (object1, object2, radius1, radius2)

#function for returning relative velocity magnitude given two vectors' magnitude & heading
def relative_velocity_mag(va, vb, theta_a, theta_b):
    vrx = va*np.cos(theta_a) - vb*np.cos(theta_b)
    vry = va*np.sin(theta_a) - vb*np.sin(theta_b)
    Vr = np.sqrt(vrx**2 + vry**2)
    return(Vr, vrx, vry)


def get_new_heading(radius2, distance, x, y, p, q, theta2, v1, v2, turnback, turn_to_original, is_left_turn, phir):
    '''angles: 
    alpha = angle between relative velocity vector and LoS vector
    beta = angle between new minimum heading that avoids collision and LoS vector
    mu = angle between relative velocity vector and new minimum heading
    phir0 = initial relative velocity heading'''
    alpha = np.arcsin(rms/distance)
    beta = np.arcsin(radius2/distance)
    mu = beta - alpha
    phir0 = arctan_new((q-y), (p-x))
    is_vr0_left_of_b = is_vector_on_left(phir, phir0)

    #phirx = new relative velocity heading, determines whether to turn left or right
    if is_left_turn:
        phirx = phir0 + beta
    else:
        phirx = phir0 - beta

    #turnback is a boolean that denotes a heading change during t2, when the aircraft is returning to its original path
    if turnback:
        if is_vr0_left_of_b:
            if is_left_turn:
                phirx = phirx - 2 * mu
            else:
                phirx = phirx + 2*(alpha + beta) 
        else:
            if is_left_turn:
                phirx = phirx - 2*(alpha + beta)
            else:
                phirx = phirx + 2 * mu
    
    #returning to original path
    if turn_to_original:
        if is_vr0_left_of_b:
            if is_left_turn:
                phirx = phirx - mu
            else:
                phirx = phirx + alpha + beta
        else:
            if is_left_turn:
                phirx = phirx - alpha - beta
            else:
                phirx = phirx + mu
    #converts appropriate relative heading required to heading change of aircraft A
    phira = phirx + np.arcsin((v2/v1) * np.sin(theta2-phirx))
    return (phira, alpha, mu, is_vr0_left_of_b)

#gets the distance between two points
def get_distance(xf1, xf2, yf1, yf2):
    t_distance = np.sqrt((xf1-xf2)**2 + (yf1-yf2)**2)
    return (t_distance, xf1, xf2, yf1, yf2)

#ode solver function: returns all points of a trajectory given initial velocity, timeframe, position, and heading
def ode_solver(t, theta, x0, y0, v):
    def dxdt(x, t, theta):
        return v * np.cos(theta)

    def dydt(y, t, theta):
        return v * np.sin(theta)

    xf = odeint(dxdt, y0=x0, t=t, tfirst=True, args=(theta,))
    yf = odeint(dydt, y0=y0, t=t, tfirst=True, args=(theta,))
    return (xf.flatten(), yf.flatten())

#calculates the predicted time to first loss (time to collision without heading change)
def time_to_first_loss(xa, ya, xb, yb, v1, v2, theta1, theta2, min_separation_nmi):
    Vr, vrx, vry = relative_velocity_mag(v1, v2, theta1, theta2)
    original_distance = np.sqrt((xa - xb)**2 + (ya - yb)**2)
    cosalpha = (vrx * (xb-xa) + vry * (yb-ya))/((Vr)*original_distance)
    sinalpha = np.sqrt(1 - np.power(cosalpha, 2))
    rms = float(original_distance * sinalpha)
    original_dfl = np.abs((original_distance * cosalpha) - np.sqrt(min_separation_nmi**2-rms**2))
    original_tfl = original_dfl/Vr
    return original_tfl

#assumption that both theta1 and theta2 are in the range [0, 2pi]
#function that determines whether the relative velocity vector is to the left or right of the LoS vector
def is_vector_on_left(theta1, theta2):
    if theta1 > theta2:
        if theta1-theta2 < np.pi:
            return True
        return False
    else:
        if theta2-theta1 < np.pi:
            return False
        return True

#initializes plot
fig, ax = plt.subplots(figsize=(8,8))

'''
is_left_turn: whether to do a left or right turn to avoid collision
min_separation_nmi: minimum allowable distance between two aircraft
tolerance: error
time_to_resolve: number of seconds before first loss until heading change'''
is_left_turn = True
#is_left_turn = False
min_separation_nmi = 100
tolerance = 0.00001
time_to_resolve = 10



#constants (velocity and theta)

#generates a random conflict scenario
r1 = random.random()
theta0 = np.radians(random.randint(0, 360))
theta_1 = np.radians(random.randint(0, 360))
theta_2 = np.radians(random.randint(0, 360))

v1 = 120
v2 = 100

#collision coordinates
xc = 0
yc = 0
pc = min_separation_nmi * r1 * np.cos(theta0)
qc = min_separation_nmi * r1 * np.sin(theta0)


#starting coordinates
tc = 60
x0 = v1 * tc * np.cos(theta_1) + xc
y0 = v1 * tc * np.sin(theta_1) + yc
p0 = v2 * tc * np.cos(theta_2) + pc
q0 = v2 * tc * np.sin(theta_2) + qc
theta1 = np.pi + theta_1
theta2 = np.pi + theta_2

print(f'x0: {x0:.2f},  y0: {y0:.2f}, p0: {p0:.2f}, q0: {q0:.2f}')
print(f'theta1: {np.degrees(theta1)}, theta2: {np.degrees(theta2)}')

#relative velocity vector components
N = v1 * np.sin(theta1) - v2 * np.sin(theta2)
D = v1 * np.cos(theta1) - v2 * np.cos(theta2)
Vr = np.sqrt(np.power(N, 2) + np.power(D, 2))
phir = arctan_new(N, D)

#initializing text display of distance variables
text = ax.text(0.5, 1.11, '', transform=ax.transAxes, ha='center')
record_min_distance = ax.text(0.5, 1.08, '', transform=ax.transAxes, ha='center')
min_distance = 10000
dmpc = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
t_conflict = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center')
rms_text = ax.text(0.5, -0.12, '', transform=ax.transAxes, ha='center')

#rigid calculations
original_distance = np.sqrt((x0 - p0)**2 + (y0 - q0)**2)
cosalpha = (D * (p0-x0) + N * (q0-y0))/((Vr)*original_distance)
sinalpha = np.sqrt(1 - np.power(cosalpha, 2))
rms = original_distance * sinalpha
original_alpha = np.arcsin(rms/original_distance)
original_beta = np.arcsin(min_separation_nmi/original_distance)
original_mu = original_beta - original_alpha

#determines if there even is a conflict
if rms + tolerance < min_separation_nmi and cosalpha > 0:
    conflict = True
else:
    conflict = False

#plotting: 
'''o = small circle
zorder = order of objects (larger number is in front)'''

#plotting 4 circles: 2 as aircraft, 2 as min allowable radius
object1 = plt.Circle((x0,y0), 1, color='blue', zorder=2, label=f'v = {v1} m/s')
object2 = plt.Circle((p0,q0), 1, color='purple', zorder=2, label=f'v = {v2} m/s')
radius1 = plt.Circle((x0,y0), min_separation_nmi, color='black', zorder=1, alpha=0.3)
radius2 = plt.Circle((p0,q0), min_separation_nmi, color='black', zorder=1, alpha=0.3)

ax.add_patch(object1)
ax.add_patch(object2)
ax.add_patch(radius1)
ax.add_patch(radius2)

#dmpc is predicted minimum distance
dmpc.set_text(f'Predicted min distance (without heading change): {rms:.2f} nmi')

# Time intervals
a = 20
fps = 5
total_time_s = 100

xai = x0
yai = y0
xbi = p0
ybi = q0

x1a = []
y1a = []
x1b = []
y1b = []

t1 = []
'''in timesteps of 1 second, checks to see if aircraft is time_to_resolve seconds away from first loss. 
if no, both aircraft continue on their original path. if yes, the t1 segment ends and t2 begins (heading change on a)'''
for i in range(total_time_s):
    tfl_i = time_to_first_loss(xai, yai, xbi, ybi, v1, v2, theta1, theta2, min_separation_nmi)
    if tfl_i > time_to_resolve:
        t_test = np.linspace(i, i+1, fps, endpoint=True)
        for j in ode_solver(t_test, theta1, xai, yai, v1)[0]:
            x1a.append(j)
        for k in ode_solver(t_test, theta1, xai, yai, v1)[1]:
            y1a.append(k)
        for l in ode_solver(t_test, theta2, xbi, ybi, v2)[0]:
            x1b.append(l)
        for m in ode_solver(t_test, theta2, xbi, ybi, v2)[1]:
            y1b.append(m)
        xai = x1a[-1]
        yai = y1a[-1]
        xbi = x1b[-1]
        ybi = y1b[-1]
        for n in t_test:
            t1.append(n)

    else:
        break
print(f't1 heading: {np.degrees(theta1):.2f}')

t2 = np.linspace(t1[-1], total_time_s, (total_time_s - int(t1[-1])) * fps, endpoint=True)

# Calculate distance and new heading angle
# initial distance = distance between two aircraft at the time of heading change
initial_distance = get_distance(x1a[-1], x1b[-1], y1a[-1], y1b[-1])[0]
new_heading_angle, alpha, mu, is_vr0_left_of_b= get_new_heading(min_separation_nmi, initial_distance, x1a[-1], y1a[-1], x1b[-1], y1b[-1], theta2, v1, v2, False, False, is_left_turn, phir)
print(f't2 heading: {np.degrees(new_heading_angle):.2f}')


Vr_star = relative_velocity_mag(v1, v2, new_heading_angle, theta2)[0]

# Calculate touchback time (time from the heading change to the moment where returning to original path is viable)
if is_vr0_left_of_b:
    if is_left_turn:
        ttb, dtb = turnback_time(initial_distance, alpha, mu, Vr_star)
    else:
        ttb, dtb = turnback_time2(initial_distance, alpha, mu, Vr_star)
else:
    if is_left_turn:
        ttb, dtb = turnback_time2(initial_distance, alpha, mu, Vr_star)
    else:
        ttb, dtb = turnback_time(initial_distance, alpha, mu, Vr_star)

#adjusts the total time in case it is too small
if t1[-1]+ttb+ttb > total_time_s:
    total_time_s = total_time_s + 30 + (t1[-1]+ttb+ttb - total_time_s)
t2 = np.linspace(t1[-1], t1[-1]+ttb, (int(ttb*fps)), endpoint=True)

# Solve ODEs for second segments
x2a, y2a = ode_solver(t2, new_heading_angle, x1a[-1], y1a[-1], v1)
x2b, y2b = ode_solver(t2, theta2, x1b[-1], y1b[-1], v2)

# Calculate return to flight-plan route
turnback_angle= get_new_heading(min_separation_nmi, initial_distance, x1a[-1], y1a[-1], x1b[-1], y1b[-1], theta2, v1, v2, True, False, is_left_turn, phir)[0]

t3 = np.linspace(t1[-1]+ttb, t1[-1]+2*ttb, (int((ttb)*fps)), endpoint=True)
print(f't3 heading: {np.degrees(turnback_angle):.2f}')

# use ode to calculate flight path for t3 (turnback)
x3a, y3a = ode_solver(t3, turnback_angle, x2a[-1], y2a[-1], v1)
x3b, y3b = ode_solver(t3, theta2, x2b[-1], y2b[-1], v2)

#calculate return to original path
t4 = np.linspace(t1[-1]+ttb+ttb, total_time_s, (int((total_time_s-(t1[-1]+ttb+ttb))*fps)), endpoint=True)
return_original_heading = get_new_heading(min_separation_nmi, initial_distance, x1a[-1], y1a[-1], x1b[-1], y1b[-1], theta2, v1, v2, False, True, is_left_turn, phir)[0]

x4a, y4a = ode_solver(t4, return_original_heading, x3a[-1], y3a[-1], v1)
x4b, y4b = ode_solver(t4, theta2, x3b[-1], y3b[-1], v2)
print(f't4 heading: {np.degrees(return_original_heading):.2f}')

#converting seconds to frames for animation
t1_num_frames = len(t1)
t1_frames = []
for i in range(t1_num_frames):
    t1_frames.append(i)

t2_num_frames = len(t2)
t2_frames = []
for i in range(t2_num_frames):
    t2_frames.append(i+t1_frames[-1]+1)

t3_num_frames = len(t3)
t3_frames = []
for i in range(t3_num_frames):
    t3_frames.append(i+t2_frames[-1]+1)

t4_num_frames = len(t4)
t4_frames = []
for i in range(t4_num_frames):
    t4_frames.append(i+t3_frames[-1]+1)

# Combine results
xa = np.concatenate((x1a, x2a, x3a, x4a))
ya = np.concatenate((y1a, y2a, y3a, y4a))
xb = np.concatenate((x1b, x2b, x3b, x4b))
yb = np.concatenate((y1b, y2b, y3b, y4b))


#setting axes
#x limit is the maximum of either the x or y position
max_limit = max(max(xa), max(xb), max(ya), max(yb))
min_limit = min(min(xa), min(xb), min(ya), min(yb))

ax.set(xlim=[min_limit, max_limit], 
       ylim=[min_limit, max_limit], 
       xlabel='x-position (nmi)', ylabel='y-position (nmi)')
ax.legend()
plt.grid()

#frames = length of animation, interval = ms between frames
# calling animation function to plot trajectory
ani = animation.FuncAnimation(fig=fig, func=update, frames=int(total_time_s*fps), interval=20, blit=False)
plt.title('Top View of 2D Trajectory Conflict')
plt.show()