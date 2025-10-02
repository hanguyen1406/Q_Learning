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
gA = 0
# Kích thước rời rạc hóa
q_table_size = [10, 10, 4, 10, 4] + [4, 4, 4, 4] + [4]  # = 10 chiều state
num_actions = 4

# Khởi tạo Q-table
Q = np.zeros((q_table_size), dtype=np.float16)
nStates = 100 * 4 * 10 * 4 * (4**4) * 2  # x,y,θ,dt,Δθ,4rays,gA
learningRate = 0.1    # learning rate
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
    res = {0: 0, 90: 1, 180: 2, 270: 3}
    return res[q * 90]

def world_to_cell(x, y):
    # chia cho kích thước cell
    cx = x // 0.5
    cy = y // 0.5
    # làm tròn: 0.24 -> 0, -0.25 -> -1
    return [int(cx), int(cy)]

def compute_reward(state):
    xt, yt, theta, dt, dtheta, *scans, g = state

    # Nếu va chạm (cảm biến = 0 hoặc nhỏ hơn ngưỡng)
    if min(scans) == 0:
        return -100, False
    
    # Nếu đã đến đích A và đang tìm A
    # if dt < 0.2 and g == 0:
    #     print(f"Đã đến đích A! Chuyển sang tìm đích B...")
    #     return +100, True
    
    # Nếu đã đến đích B và đang tìm B  
    if dt == 0:
        print(f"Đã đến đích B! Hoàn thành nhiệm vụ!")
        return +100, True

    # Nếu đang tiến gần đích (so sánh với step trước)
    if hasattr(compute_reward, "last_d"):
        reward = 5 * (compute_reward.last_d - dt)
    else:
        reward = 0
    compute_reward.last_d = dt

    # Phạt mỗi bước nhỏ
    reward += -0.01

    return reward, False



def get_state():
    # --- Pose robot ---
    pos = sim.getObjectPosition(robot, -1)   # [x, y, z]
    ori = sim.getObjectOrientation(robot, -1)  # [alpha, beta, gamma] rad
    xt, yt = pos[0], pos[1]
    θt = ori[2]   # góc yaw

    # --- Tính khoảng cách đến đích hiện tại ---
    posA = sim.getObjectPosition(dichB, -1)  # Vị trí đích A (Cuboid)
    # posB = sim.getObjectPosition(dichB, -1)  # Vị trí đích B (handtruck)
    
    # if gA == 0:
    target_pos = posA
    dx, dy = target_pos[0] - xt, target_pos[1] - yt
    dt = np.sqrt(dx**2 + dy**2)
    # Góc từ robot đến đích A
    angleToTarget = np.arctan2(dy, dx)
    # else: 
    #     target_pos = posB
    #     dx, dy = target_pos[0] - xt, target_pos[1] - yt
    #     dt = np.sqrt(dx**2 + dy**2)
    #     # Góc từ robot đến đích B
    #     angleToTarget = np.arctan2(dy, dx)

    # --- Sai lệch hướng Δθt so với hướng đến đích ---
    Δθt = angleToTarget - θt
    Δθt = np.arctan2(np.sin(Δθt), np.cos(Δθt))  # chuẩn hóa [-pi, pi]

    # --- Scan 4 tia ---
    scans = []
    for name, h in ray_handles.items():
        # print(name)
        # dis = read_ray(h)
        dis = round(read_ray(h))
        scans.append(dis)

    st = world_to_cell(xt, yt) + [quantize_angle(θt), round(dt), quantize_angle(Δθt)] + scans + [gA]
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

def step(action):
    # 1. Cho robot di chuyển 1 bước theo action
    move_cell(action)

    # 2. Lấy state mới từ simulator
    next_state = get_state()   # [xt, yt, θt, dt, Δθt, scan1..n]

    # 3. Tính reward
    reward, done = compute_reward(next_state)

    return next_state, reward, done


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

def episode_loop():
    global v_epsilon, Q, actions, gA,max_ep_reward
    for ep in range(numEps):
        resetRobot()
        print("Eps = ", ep)
        done = False
        current_state = get_state()
        ep_reward = 0
        ep_start_state = current_state
        action_list = []
        while not done:
            
            if np.random.random() > v_epsilon:
                # Lấy argmax Q value của current_state
                action = np.argmax(Q[current_state])
            else:
                action = np.random.randint(0, num_actions)

            action_list.append(action)

            next_real_state, reward, done  = step(action=actions[action])
            print("Next real state:", next_real_state, "Reward:", reward, "Done:", done)
            ep_reward += reward
            if done:
                #kiểm tra đến đích A hoặc B chưa
                # if gA == 0 and next_real_state[3] < 0.2:
                #     gA = 1
                #     print("Đến đích A")
                # elif gA == 1 and next_real_state[3] < 0.2:
                #     gA = 2
                #     print("Đến đích B")
                # print("Đến đích")
                if ep_reward > max_ep_reward:
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
                    max_start_state = ep_start_state
            else:
                # Update Q value cho (current_state, action)
                # print("current state:", current_state)
                # print("action:", action)
                current_q_value = Q[current_state + (action,)]
                new_q_value = (1 - learningRate) * current_q_value + learningRate * (reward + discountFactor * np.max(Q[next_real_state]))
                Q[current_state + (action,)] = new_q_value
                current_state = next_real_state


print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)

# ===== traning =====
resetRobot()
# for d in ["down", "down", "down", "up", "up", "right"]:
#     move_cell(d)
#     state = get_state()
#     print(d, ":", state)
#     time.sleep(2)

episode_loop()



time.sleep(2)
sim.stopSimulation()
