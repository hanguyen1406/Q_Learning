# nav_ab_qlearning_4rays_abs_orient.py
import math, time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ===== cấu hình môi trường =====
LEFT, RIGHT = '/PioneerP3DX/leftMotor', '/PioneerP3DX/rightMotor'
WHEEL_RADIUS = 0.07     # m
BASELINE     = 0.30     # m
LIN_SPEED    = 0.25     # m/s
MAP_RES      = 0.5      # 1 cell = 0.5 m
heading = 0.0  # robot hướng ban đầu (rad, trục Z)

# ===== Cấu hình qlearning =====


# ===== Connect =====
client = RemoteAPIClient()
sim = client.getObject('sim')
robot = sim.getObject('/PioneerP3DX')
left  = sim.getObject(LEFT)
right = sim.getObject(RIGHT)



def resetRobot():
    global heading
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
    move_cell(d)
    time.sleep(2)



time.sleep(2)
sim.stopSimulation()
