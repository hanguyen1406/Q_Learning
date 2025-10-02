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
actions = ["up", "down", "left", "right"]
# ===== Cấu hình qlearning =====
last_d = None # khoảng cách đến đích ở step trước
# Kích thước rời rạc hóa
q_table_size = [36, 10] + [4, 4, 4, 4] + [4]  # = 10 chiều state
num_actions = 4
max_step = 10000
# Khởi tạo Q-table
Q = None
learningRate = 0.5    # learning rate
discountFactor = 0.9  # discount factor
v_epsilon = 0.9
numEps = 1000
previS = None

max_ep_reward = -999
max_ep_action_list = []
max_start_state = None

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

def save_Qtable(Qtable, filename='Qtable.npy'):
    np.save(filename, Qtable)
    print("Q-table đã lưu!")

def load_Qtable(filename='Qtable.npy'):
    try:
        Qtable = np.load(filename)
        print("load Q-table!")
    except FileNotFoundError:
        Qtable = np.zeros((q_table_size))  # Nếu không có file, khởi tạo bảng Q mặc định
        print("không thấy Q-table, tạo mới!")
    return Qtable

def read_ray(handle):
    res = sim.checkProximitySensor(handle, sim.handle_all)
    detected, dist = res[0], res[1]
    if detected > 0 and dist and dist > 0:
        return dist
    else:
        return MAX_RANGE
    
def quantize_angle(theta_rad):
    deg = (math.degrees(theta_rad) % 360 + 360) % 360
    idx = int(round(deg / 10)) % 36
    return idx

def discretize(value, low=-5.0, high=5.0, step=0.1):
    rounded = round(value, 1)
    idx = int((rounded - low) / step)
    return idx

def world_to_cell(x, y):
    return [discretize(x, 1), discretize(y, 1)]

def compute_reward(state):
    theta, dt, dtheta, *scans, g = state
    global last_d
    if min(scans) == 0:
        return -100, False
    if dt == 0:
        print(f"Đã đến đích B! Hoàn thành nhiệm vụ!")
        return +100, True
    if last_d is None:
        reward = 0
    else:
        reward = 5 * (last_d - dt)
    last_d = dt

    reward += -0.01
    return reward, False

def get_state():
    pos = sim.getObjectPosition(robot, -1)   # [x, y, z]
    ori = sim.getObjectOrientation(robot, -1)  # [alpha, beta, gamma] rad
    xt, yt = pos[0], pos[1]
    θt = ori[2]   # góc yaw
    posA = sim.getObjectPosition(dichB, -1)  # Vị trí đích A (Cuboid)
    target_pos = posA
    dx, dy = target_pos[0] - xt, target_pos[1] - yt
    dt = np.sqrt(dx**2 + dy**2)
    angleToTarget = np.arctan2(dy, dx)
   
    Δθt = angleToTarget - θt
    Δθt = np.arctan2(np.sin(Δθt), np.cos(Δθt))  # chuẩn hóa [-pi, pi]
    scans = []
    for name, h in ray_handles.items():
        dis = round(read_ray(h))
        scans.append(dis)
    st = [quantize_angle(θt), round(dt)] + scans
    return tuple(st)



def resetRobot():
    global heading,gA
    gA = 0
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped: pass
    sim.setObjectPosition(robot, -1, [-2.25, -2.25, 0.15])
    sim.setObjectOrientation(robot, -1, [0, 0, 0])
    heading = 0.0
    sim.setJointTargetVelocity(left, 0)
    sim.setJointTargetVelocity(right, 0)
    sim.setStepping(True)
    sim.startSimulation()

def step(action):
    move_cell(action)
    next_state = get_state()   # [xt, yt, θt, dt, Δθt, scan1..n]
    reward, done = compute_reward(next_state)
    return next_state, reward, done

def _run_for(dt, v, w=0.0):
    wL = (v - 0.5 * w * BASELINE) / WHEEL_RADIUS
    wR = (v + 0.5 * w * BASELINE) / WHEEL_RADIUS
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
    small_angle = math.radians(20)
    w_turn = 0.6
    t_turn = small_angle / w_turn

    if direction == "up":
        _run_for(t_fwd, LIN_SPEED, 0.0)
    elif direction == "down":
        _run_for(t_fwd, -LIN_SPEED, 0.0)
    elif direction == "left":
        _run_for(t_turn, 0.0, +w_turn)   # quay trái
        heading += small_angle
    elif direction == "right":
        _run_for(t_turn, 0.0, -w_turn)   # quay phải
        heading -= small_angle

def episode_loop():
    global v_epsilon, Q, actions, gA, max_ep_reward, max_step
    for ep in range(numEps):
        resetRobot()
        print("Eps = ", ep)
        done = False
        current_state = get_state()
        ep_reward = 0
        ep_start_state = current_state
        action_list = []
        t = 0
        while not done and t < max_step:
            t += 1
            if np.random.random() > v_epsilon:
                action = np.argmax(Q[current_state])
            else:
                action = np.random.randint(0, num_actions)
            action_list.append(action)
            next_real_state, reward, done  = step(action=actions[action])
            ep_reward += reward
            if done:
                if ep_reward > max_ep_reward:
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
                    max_start_state = ep_start_state
            else:
                current_q_value = Q[current_state + (action,)]
                new_q_value = (1 - learningRate) * current_q_value + learningRate * (reward + discountFactor * np.max(Q[next_real_state]))
                Q[current_state + (action,)] = new_q_value
                current_state = next_real_state
            t += 1
        save_Qtable(Q)

print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)

# ===== traning =====
Q = load_Qtable()
resetRobot()
# for d in ["up", "left", "up", "up", "left", "right"]:
#     move_cell(d)
#     state = get_state()
#     print(d, ":", state)
#     time.sleep(2)

episode_loop()



time.sleep(2)
sim.stopSimulation()
