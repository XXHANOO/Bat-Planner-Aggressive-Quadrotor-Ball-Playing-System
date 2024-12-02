import numpy as np
import matplotlib.pyplot as plt

# Kalman filter implementation
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimation error covariance
        self.x = x0  # Initial state estimate

    def predict(self, u):
        # Prediction step
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        # Update step
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)  # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

# Simulation parameters
dt = 0.1  # Time step
time = np.arange(0, 20, dt)  # Simulation time

# Define the system
A = np.array([[1, dt], [0, 1]])  # State transition matrix
B = np.array([[0], [dt]])        # Control matrix
H = np.array([[1, 0]])           # Measurement matrix
Q = np.eye(2) * 0.1              # Process noise covariance
R = np.array([[0.5]])            # Measurement noise covariance
P = np.eye(2)                    # Initial estimation error covariance
x0 = np.array([[0], [1]])        # Initial state [position, velocity]

# Initialize Kalman filter
kf = KalmanFilter(A, B, H, Q, R, P, x0)

# Simulate true trajectory and noisy measurements
true_positions = []
measurements = []
estimated_positions = []

x_true = x0.copy()
for t in time:
    # Simulate true system dynamics
    u = np.array([[0]])  # No control input
    x_true = A @ x_true + B @ u + np.random.multivariate_normal([0, 0], Q).reshape(-1, 1)

    # Simulate noisy measurement
    z = H @ x_true + np.random.normal(0, R[0, 0], (1, 1))

    # Kalman filter prediction and update
    kf.predict(u)
    kf.update(z)

    # Store results
    true_positions.append(x_true[0, 0])
    measurements.append(z[0, 0])
    estimated_positions.append(kf.x[0, 0])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, true_positions, label="True Position", linestyle="--")
plt.plot(time, measurements, label="Measurements", alpha=0.6)
plt.plot(time, estimated_positions, label="Kalman Filter Estimate")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Kalman Filter for Quadrotor Trajectory Estimation")
plt.grid()
plt.show()
