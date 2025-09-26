import numpy as np
import random
from controller import Robot
import time

def convert_sensor_to_distance(value):
    if value > 0:
        k = 5000
        return k / value
    return float('inf')

def classify_distance(distance):
    if 15 <= distance < 30:
        return 0
    elif 30 <= distance < 65:
        return 1
    elif distance >= 65:
        return 2
    return -1

def reward(states, action):
    r1 = 0
    r2 = 0
    r3 = 0

    if action == 1 or action == 2:
        r1 = -1
    elif action == 0:
        r2 = +3
    if -1 in states:
        r3 = -15

    reward = r1 + r2 + r3
    return reward

def get_sensor_distances(robot):
    timestep = int(robot.getBasicTimeStep())
    prox_sensors = []
    for i in range(8):
        sensor = robot.getDevice(f'ps{i}')  # Thay 'ps{i}' với getDevice thay cho getMotor
        sensor.enable(timestep)
        prox_sensors.append(sensor)
    states = []
    for sensor in prox_sensors:
        raw_value = sensor.getValue()
        distance = convert_sensor_to_distance(raw_value)
        state = classify_distance(distance)
        states.append(state)
    return states
    
def epsilon_greedy_policy(Qtable, state, epsilon):
    # Kiểm tra nếu tất cả các cảm biến đều có trạng thái là 2
    if all(s == 2 for s in state):  # Tất cả các trạng thái đều là 2
        return 0  # Dừng lại (action = 0)

    # Nếu không, thực hiện khám phá hoặc khai thác theo epsilon-greedy
    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        state_indices = tuple(state)  # state có thể chứa giá trị từ 0, 1, 2 hoặc -1
        action = np.argmax(Qtable[state_indices])  # Tối ưu hóa
    else:
        action = random.randint(0, 2)  # Khám phá ngẫu nhiên từ không gian hành động

    return action

def perform_action(robot, action):
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    
    if action == 0:  # Đi thẳng
        left_motor.setVelocity(6.28)
        right_motor.setVelocity(6.28)
    elif action == 1:  # Quay trái
        left_motor.setVelocity(-6.28)  # Điều chỉnh tốc độ quay trái
        right_motor.setVelocity(6.28)  # Điều chỉnh tốc độ quay trái
    elif action == 2:  # Quay phải
        left_motor.setVelocity(6.28)  # Điều chỉnh tốc độ quay phải
        right_motor.setVelocity(-6.28)  # Điều chỉnh tốc độ quay phải
    elif action == 3:  # Quay lùi lại (lùi 3 giây)
        left_motor.setVelocity(-6.28)  # Lùi
        right_motor.setVelocity(-6.28)  # Lùi

def save_Qtable(Qtable, filename='Qtable.npy'):
    np.save(filename, Qtable)
    print("Q-table saved!_Q_learning")

def load_Qtable(filename='Qtable.npy'):
    try:
        Qtable = np.load(filename)
        print("Q-table loaded!")
    except FileNotFoundError:
        Qtable = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3))  # Nếu không có file, khởi tạo bảng Q mặc định
        print("Q-table not found, initialized new one!")
    return Qtable

def update_Qtable(Qtable, state, action, reward_value, next_state, alpha=0.1, gamma=0.9):
    state_indices = tuple(state)
    next_state_indices = tuple(next_state)

    # Tính toán giá trị Q mới bằng công thức Q-learning
    current_q = Qtable[state_indices + (action,)]
    max_next_q = np.max(Qtable[next_state_indices])  # Tối đa các giá trị Q của các hành động ở trạng thái tiếp theo

    # Cập nhật giá trị Q theo công thức Q = Q + alpha * (reward + gamma * max(Q') - Q)
    Qtable[state_indices + (action,)] = current_q + alpha * (reward_value + gamma * max_next_q - current_q)

def run_robot(robot):
    timestep = int(robot.getBasicTimeStep())
    Qtable = load_Qtable()  # Tải bảng Q từ file

    epsilon = 1  # Xác suất khám phá
    alpha = 0.5  # Tốc độ học
    gamma = 0.9  # Hệ số chiết khấu
    epsilon = max(0.1, epsilon * 0.99)
    desired_position = (-0.194067, 0.151123, -0.0579739)  # Tọa độ mong muốn sau khi reset
    
    while robot.step(timestep) != -1:
        states = get_sensor_distances(robot)
        action = epsilon_greedy_policy(Qtable, states, epsilon)  # Hành động chọn theo chính sách epsilon-greedy
        print(f"Action taken: {action}")
        
        # Áp dụng hàm thưởng cho các trạng thái và hành động
        reward_value = reward(states, action)
        print(f"Reward: {reward_value}")

        print(f"Sensor states: {states}")
        print("-" * 30)
        
        # Thực hiện hành động với mô-đun động cơ
        perform_action(robot, action)

        # Lấy trạng thái mới sau khi thực hiện hành động
        next_states = get_sensor_distances(robot)

        # Cập nhật bảng Q
        update_Qtable(Qtable, states, action, reward_value, next_states, alpha, gamma)

        save_Qtable(Qtable)
        
        if -1 in states:  
            print("Collision detected! Reversing for 1cho tô seconds.")
            perform_action(robot, 3)  
            for _ in range(int(1000 / timestep)): 
                robot.step(timestep)  
                
            continue  

if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)
