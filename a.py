from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Kết nối đến CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')

# Lấy handle 4 sensor
hUp    = sim.getObject('/PioneerP3DX/up')     # front
hDown  = sim.getObject('/PioneerP3DX/down')   # back
hLeft  = sim.getObject('/PioneerP3DX/left')   # left
hRight = sim.getObject('/PioneerP3DX/right')  # right

MAX_RANGE = 3.0  # khớp với Ray length trong properties

def read_ray(handle):
    # checkProximitySensor trả về: detected, distance, detectedPoint, detectedObject, normalVector
    res = sim.checkProximitySensor(handle, sim.handle_all)
    detected, dist = res[0], res[1]
    if detected > 0 and dist and dist > 0:
        return dist
    else:
        return MAX_RANGE

# Bắt đầu mô phỏng
sim.setStepping(True)
sim.startSimulation()

try:
    while sim.getSimulationState() != sim.simulation_stopped:
        rFront = read_ray(hUp)
        rBack  = read_ray(hDown)
        rLeft  = read_ray(hLeft)
        rRight = read_ray(hRight)

        print(f"Up={rFront:.2f}  Right={rRight:.2f}  Left={rLeft:.2f}  Down={rBack:.2f}")

        sim.step()
finally:
    sim.stopSimulation()
