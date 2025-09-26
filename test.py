# nav_ab_qlearning_4rays.py
import math, random, numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ====== Config ======
MODE = "demo"  # "train" hoặc "demo"
LEFT, RIGHT = '/PioneerP3DX/leftMotor', '/PioneerP3DX/rightMotor'
RAYS = {
    "front": '/PioneerP3DX/up',
    "back":  '/PioneerP3DX/down',
    "left":  '/PioneerP3DX/left',
    "right": '/PioneerP3DX/right'
}
GOAL_A, GOAL_B = '/Cuboid', '/handtruck'
WHEEL_SPEED, ACTIONS = 6.0, 3

# thresholds
RTH_NEAR, RTH_MED, RTH_FAR, MAX_RANGE = 0.15, 0.30, 0.65, 3.0
GOAL_BINS, GOAL_REACHED = [0.20, 0.50, 1.00], 0.20
BEARING_BINS = [-60, -15, 15, 60]

# Q-learning
ALPHA, GAMMA = 0.5, 0.9
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.1, 0.995
MAX_STEPS, N_EP = 1200, (200 if MODE=="train" else 1)
QFILE = 'Qtable_nav_ab.npy'

# ====== Helpers ======
def bin_ray(d): return 0 if d<RTH_NEAR else 1 if d<RTH_MED else 2 if d<RTH_FAR else 3
def bin_dist(d): return next((i for i,t in enumerate(GOAL_BINS) if d<t), len(GOAL_BINS))
def bin_bear(b):
    if b<BEARING_BINS[0]: return 0
    if b<BEARING_BINS[1]: return 1
    if b<=BEARING_BINS[2]: return 2
    if b<=BEARING_BINS[3]: return 3
    return 4
def wrap(a): return (a+180)%360-180
def dist(x1,y1,x2,y2): return math.hypot(x1-x2,y1-y2)

# ====== Connect ======
client=RemoteAPIClient(); sim=client.getObject('sim')
left,right=sim.getObject(LEFT),sim.getObject(RIGHT)
rH={k:sim.getObject(v) for k,v in RAYS.items()}
robot,goalA,goalB=sim.getObject('/PioneerP3DX'),sim.getObject(GOAL_A),sim.getObject(GOAL_B)

# ====== Q-table ======
STATE_DIMS=(4,4,4,4,4,5); Qshape=STATE_DIMS+(ACTIONS,)
try: Q=np.load(QFILE); assert Q.shape==Qshape
except: Q=np.zeros(Qshape,np.float32)

# ====== Sensors ======
def read_ray(h):
    d=sim.checkProximitySensor(h,sim.handle_all)[1]
    return d if d and d>0 else MAX_RANGE
def pose():
    p,o=sim.getObjectPosition(robot,-1),sim.getObjectOrientation(robot,-1)
    return p[0],p[1],o[2]
def goal_pos(h): p=sim.getObjectPosition(h,-1); return p[0],p[1]

def get_state(goal):
    rays=[read_ray(rH['front']),read_ray(rH['right']),
          read_ray(rH['left']),read_ray(rH['back'])]
    rb=[bin_ray(r) for r in rays]
    rx,ry,yaw=pose(); gx,gy=goal_pos(goal)
    dx,dy=gx-rx,gy-ry
    rel=wrap(math.degrees(math.atan2(dy,dx))-math.degrees(yaw))
    return tuple(rb+[bin_dist(math.hypot(dx,dy)),bin_bear(rel)]),rays,rel

# ====== Actions ======
def act(a):
    if a==0: vl,vr=WHEEL_SPEED,WHEEL_SPEED
    elif a==1: vl,vr=-WHEEL_SPEED,WHEEL_SPEED
    else: vl,vr=WHEEL_SPEED,-WHEEL_SPEED
    sim.setJointTargetVelocity(left,vl); sim.setJointTargetVelocity(right,vr)

# ====== Reward ======
def rew(s,r,rel,goal):
    R = -0.5                          # step cost thấp thôi
    if 0 in s[:4]: R -= 20.0          # có tia rất gần -> phạt nặng
    if s[0] <= 1: R -= 5.0            # trước mặt không an toàn
    # thưởng hướng về goal (mượt dần theo |rel|)
    R += max(0, 3.0 - abs(rel)/20.0)  # ~3 khi |rel|=0; 0 khi |rel|>=60
    # về đích
    rx,ry,_ = pose(); gx,gy = goal_pos(goal)
    if math.hypot(rx-gx, ry-gy) < GOAL_REACHED: R += 120.0
    return R

# ====== Reset ======
def reset_to(h):
    sim.stopSimulation()
    while sim.getSimulationState()!=sim.simulation_stopped: pass

    if MODE=="train":
        # chỉ train mới teleport
        x,y=goal_pos(h)
        sim.setObjectPosition(robot,-1,[x,y,0.05])
        sim.setObjectOrientation(robot,-1,[0,0,random.uniform(-math.pi,math.pi)])

    sim.setJointTargetVelocity(left,0); sim.setJointTargetVelocity(right,0)
    sim.setStepping(True); sim.startSimulation()


# ====== Policy ======
def policy(Q,s,eps):
    return int(np.argmax(Q[s])) if random.random()>eps else random.randint(0,ACTIONS-1)

# ====== Main loop ======
eps=EPS_START if MODE=="train" else 0.0
for ep in range(N_EP):
    start,goal=(goalA,goalB) if ep%2==0 else (goalB,goalA)
    reset_to(start); steps=0
    while sim.getSimulationState()!=sim.simulation_stopped and steps<MAX_STEPS:
        s,rays,rel=get_state(goal); a=policy(Q,s,eps if MODE=="train" else 0)
        act(a); sim.step()
        s2,_,rel2=get_state(goal); R=rew(s,rays,rel,goal)
        if MODE=="train":
            cur=Q[s+(a,)]; Q[s+(a,)]=cur+ALPHA*(R+GAMMA*np.max(Q[s2])-cur)
        rx,ry,_=pose(); gx,gy=goal_pos(goal)
        if dist(rx,ry,gx,gy)<GOAL_REACHED: break
        steps+=1
    if MODE=="train":
        eps=max(EPS_MIN,eps*EPS_DECAY); np.save(QFILE,Q)
        print(f"EP{ep+1}/{N_EP} steps={steps} eps={eps:.3f}")
    else: print(f"[DEMO] steps={steps}"); break
sim.stopSimulation()
