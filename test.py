# nav_ab_qlearning_4rays_abs_orient.py
import math, time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ===== cấu hình môi trường =====
LEFT, RIGHT = '/PioneerP3DX/leftMotor', '/PioneerP3DX/rightMotor'
WHEEL_RADIUS = 0.07     # m
BASELINE     = 0.30     # m
LIN_SPEED    = 0.25     # m/s
MAP_RES      = 0.5      # 1 cell = 0.5 m
MAX_RANGE = 3.0  # khớp với Ray length trong properties
heading = 0.0  # robot hướng ban đầu (rad, trục Z)

# ===== Cấu hình qlearning =====
gA = 0
nStates = 100 * 4 * 10 * 4 * (4**4) * 2  # x,y,θ,dt,Δθ,4rays, gA
nActions = 4  # up, down, left, right
Q = np.zeros((nStates, nActions))
learningRate = 0.1    # learning rate
discountFactor = 0.9  # discount factor
# ===== Connect =====
client = RemoteAPIClient()
sim = client.getObject('sim')
robot = sim.getObject('/PioneerP3DX')
dichA = sim.getObject('/Cuboid')
dichB = sim.getObject('/handtruck')
left  = sim.getObject(LEFT)
right = sim.getObject(RIGHT)
ray_handles = {
    "front": sim.getObject('/PioneerP3DX/up'),
    "back":  sim.getObject('/PioneerP3DX/down'),
    "left":  sim.getObject('/PioneerP3DX/left'),
    "right": sim.getObject('/PioneerP3DX/right')
}


def read_ray(handle):
    res = sim.checkProximitySensor(handle, sim.handle_all)
    detected, dist = res[0], res[1]
    if detected > 0 and dist and dist > 0:
        return dist
    else:
        return MAX_RANGE
    
def quantize_angle(theta_rad):
    # rad → độ, đưa về [0,360)
    deg = (math.degrees(theta_rad) % 360 + 360) % 360
    # chia cho 90 và làm tròn
    q = round(deg / 90) % 4
    return q * 90

def world_to_cell(x, y):
    # chia cho kích thước cell
    cx = x // 0.5
    cy = y // 0.5
    # làm tròn: 0.24 -> 0, -0.25 -> -1
    return [cx, cy]

def get_state():
    # --- Pose robot ---
    pos = sim.getObjectPosition(robot, -1)   # [x, y, z]
    ori = sim.getObjectOrientation(robot, -1)  # [alpha, beta, gamma] rad
    xt, yt = pos[0], pos[1]
    θt = ori[2]   # góc yaw

    # --- Đích B ---
    posB = sim.getObjectPosition(dichB, -1)
    dx, dy = posB[0] - xt, posB[1] - yt
    dt = np.sqrt(dx**2 + dy**2)

    # --- Sai lệch hướng Δθt so với vector A→B ---
    posA = sim.getObjectPosition(dichA, -1)
    vAB = np.array([posB[0]-posA[0], posB[1]-posA[1]])
    angleAB = np.arctan2(vAB[1], vAB[0])
    Δθt = angleAB - θt
    Δθt = np.arctan2(np.sin(Δθt), np.cos(Δθt))  # chuẩn hóa [-pi, pi]

    # --- Scan 4 tia ---
    scans = []
    for name, h in ray_handles.items():
        # print(name)
        # dis = read_ray(h)
        dis = round(read_ray(h))
        scans.append(dis)

    st = world_to_cell(xt, yt) + [quantize_angle(θt), round(dt), quantize_angle(Δθt)] + scans + [gA]
    return st



def resetRobot():
    global heading
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped: pass
    sim.setObjectPosition(robot, -1, [-2.25, -2.25, 0.15])
    heading = 0.0
    sim.setJointTargetVelocity(left, 0)
    sim.setJointTargetVelocity(right, 0)
    sim.setStepping(True)
    sim.startSimulation()

def _run_for(dt, v):
    """Đi thẳng/lùi trong dt giây với tốc độ v"""
    w = 0.0
    wL = (v - 0.5*w*BASELINE) / WHEEL_RADIUS
    wR = (v + 0.5*w*BASELINE) / WHEEL_RADIUS
    sim.setJointTargetVelocity(left,  wL)
    sim.setJointTargetVelocity(right, wR)
    tend = sim.getSimulationTime() + dt
    while sim.getSimulationTime() < tend:
        sim.step()
    sim.setJointTargetVelocity(left,  0)
    sim.setJointTargetVelocity(right, 0)

def move_cell(direction: str):
    global heading
    t_fwd = (MAP_RES / LIN_SPEED) * 0.7

    if direction == "up":
        _run_for(t_fwd, LIN_SPEED)
    elif direction == "down":
        _run_for(t_fwd, -LIN_SPEED)
    elif direction == "left":
        heading += math.pi/2
        sim.setObjectOrientation(robot, -1, [0, 0, heading])
    elif direction == "right":
        heading -= math.pi/2
        sim.setObjectOrientation(robot, -1, [0, 0, heading])

# ===== Demo =====
resetRobot()
for d in ["up", "up", "left", "up", "right", "down"]:
    # ===== Demo lấy state =====
    move_cell(d)
    state = get_state()
    print(d, ":", state)
    time.sleep(2)



time.sleep(2)
sim.stopSimulation()
