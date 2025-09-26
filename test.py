# coppelia_qlearning_4rays.py
import numpy as np, random
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ====== Tham số dễ chỉnh ======
LEFT_JOINT_PATH   = '/PioneerP3DX/leftMotor'
RIGHT_JOINT_PATH  = '/PioneerP3DX/rightMotor'

# 4 tia ray gắn trong PioneerP3DX
RAY_FRONT_PATH = '/PioneerP3DX/up'     # front
RAY_BACK_PATH  = '/PioneerP3DX/down'   # back
RAY_LEFT_PATH  = '/PioneerP3DX/left'   # left
RAY_RIGHT_PATH = '/PioneerP3DX/right'  # right

ACTIONS = 3                 # 0: thẳng, 1: rẽ trái, 2: rẽ phải
WHEEL_SPEED = 6.28          # rad/s
SECTORS = 4                 # 4 tia cố định: front/right/left/back

# Ngưỡng theo mét cho phân loại khoảng cách
THRESH_NEAR   = 0.15        # < 0.15 m  -> -1 (rất gần/nguy hiểm)
THRESH_MEDIUM = 0.30        # [0.15,0.30) -> 0
THRESH_FAR    = 0.65        # [0.30,0.65) -> 1 ; >=0.65 -> 2
MAX_RANGE     = 3.00        # khớp Ray length của sensor; dùng khi không phát hiện

ALPHA = 0.5
GAMMA = 0.9
EPS_START = 1.0
EPS_MIN = 0.1
EPS_DECAY = 0.999

# ====== Utils ======
def classify_distance_m(d: float) -> int:
    """Trả về {-1,0,1,2} theo ngưỡng mét."""
    if d < THRESH_NEAR:         return -1
    if d < THRESH_MEDIUM:       return 0
    if d < THRESH_FAR:          return 1
    return 2

def reward_fn(states, action):
    r = 0
    if action in (1, 2): r -= 1      # rẽ nhẹ phạt
    if action == 0:      r += 3      # đi thẳng thưởng
    if -1 in states:     r -= 15     # rất gần vật cản
    return r

def eps_greedy(Q, s_enc, eps):
    # nếu cả 4 tia đều xa (2 -> enc=3) thì ưu tiên đi thẳng
    if all(x == 3 for x in s_enc):
        return 0
    return int(np.argmax(Q[tuple(s_enc)])) if random.random() > eps else random.randint(0, ACTIONS-1)

# ====== Kết nối CoppeliaSim ======
client = RemoteAPIClient()
sim = client.getObject('sim')

left  = sim.getObject(LEFT_JOINT_PATH)
right = sim.getObject(RIGHT_JOINT_PATH)

hFront = sim.getObject(RAY_FRONT_PATH)
hBack  = sim.getObject(RAY_BACK_PATH)
hLeft  = sim.getObject(RAY_LEFT_PATH)
hRight = sim.getObject(RAY_RIGHT_PATH)

# ====== Q-table: mỗi sector có 4 mã (enc 0..3) cho {-1,0,1,2} => shape (4,)^4 x ACTIONS
try:
    Q = np.load('Qtable.npy')
    if Q.shape != (4,)*SECTORS + (ACTIONS,):
        raise ValueError('Qtable shape mismatch')
except Exception:
    Q = np.zeros((4,)*SECTORS + (ACTIONS,), dtype=float)

# ====== Đọc 1 ray (yêu cầu sensor bật Explicit handling) ======
def read_ray(handle) -> float:
    # checkProximitySensor -> (detected, distance, detectedPoint, detectedObject, normalVector)
    detected, dist, *_ = sim.checkProximitySensor(handle, sim.handle_all)
    if detected > 0 and dist and dist > 0:
        return float(dist)
    return MAX_RANGE

# ====== Lấy state từ 4 tia ======
def ray_states_4():
    # thứ tự state: [front, right, left, back]
    front = read_ray(hFront)
    right = read_ray(hRight)
    left  = read_ray(hLeft)
    back  = read_ray(hBack)

    # phân loại sang {-1,0,1,2}
    s = [classify_distance_m(front),
         classify_distance_m(right),
         classify_distance_m(left),
         classify_distance_m(back)]
    return s

# ====== Điều khiển robot ======
def do_action(a):
    if a == 0: vl, vr =  WHEEL_SPEED,  WHEEL_SPEED
    elif a == 1: vl, vr = -WHEEL_SPEED,  WHEEL_SPEED
    elif a == 2: vl, vr =  WHEEL_SPEED, -WHEEL_SPEED
    else: vl, vr = 0.0, 0.0
    sim.setJointTargetVelocity(left,  vl)
    sim.setJointTargetVelocity(right, vr)

# ====== Main loop ======
sim.setStepping(True)
sim.startSimulation()

epsilon = EPS_START
try:
    while sim.getSimulationState() != sim.simulation_stopped:
        # đọc 4 tia & mã hoá state
        s = ray_states_4()              # [-1,0,1,2] x 4
        s_enc = [x + 1 for x in s]      # enc thành [0..3] x 4 để index Q

        # chọn hành động và thực thi
        a = eps_greedy(Q, s_enc, epsilon)
        do_action(a)

        # tiến mô phỏng
        sim.step()

        # trạng thái tiếp theo + cập nhật Q
        s2 = ray_states_4()
        s2_enc = [x + 1 for x in s2]

        R = reward_fn(s, a)
        cur = Q[tuple(s_enc)+(a,)]
        Q[tuple(s_enc)+(a,)] = cur + ALPHA*(R + GAMMA*np.max(Q[tuple(s2_enc)]) - cur)

        # lưu Q
        np.save('Qtable.npy', Q)

        # epsilon decay
        epsilon = max(EPS_MIN, epsilon*EPS_DECAY)

        # phản xạ an toàn: nếu có -1 ở bất kỳ tia nào, lùi 1 bước ngắn
        if -1 in s:
            sim.setJointTargetVelocity(left,  -WHEEL_SPEED)
            sim.setJointTargetVelocity(right, -WHEEL_SPEED)
            sim.step()
finally:
    sim.stopSimulation()
