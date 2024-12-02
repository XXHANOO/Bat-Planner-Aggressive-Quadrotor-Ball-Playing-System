% Simulation parameters
dt = 0.1; % Time step
time = 0:dt:20; % Simulation duration

% State-space matrices
A = [1 dt; 0 1]; % State transition matrix
B = [0; dt]; % Control matrix
H = [1 0]; % Measurement matrix
Q = [0.1 0; 0 0.1]; % Process noise covariance
R = 0.5; % Measurement noise covariance
P = eye(2); % Initial covariance
x = [0; 1]; % Initial state [position; velocity]

% Kalman filter storage
positions = [];
measurements = [];
estimates = [];

% Simulate dynamics and Kalman filter
for t = time
    % Simulate true position with noise
    x = A * x + B * 0 + mvnrnd([0; 0], Q)';
    true_pos = H * x;

    % Simulate measurement with noise
    z = true_pos + normrnd(0, sqrt(R));

    % Kalman filter prediction
    x_pred = A * x;
    P_pred = A * P * A' + Q;

    % Kalman filter update
    K = P_pred * H' / (H * P_pred * H' + R);
    x = x_pred + K * (z - H * x_pred);
    P = (eye(2) - K * H) * P_pred;

    % Store results
    positions = [positions; true_pos];
    measurements = [measurements; z];
    estimates = [estimates; x(1)];
end

% Plot results
figure;
plot(time, positions, '--', 'DisplayName', 'True Position');
hold on;
plot(time, measurements, '.', 'DisplayName', 'Measurements');
plot(time, estimates, '-', 'DisplayName', 'Kalman Estimate');
xlabel('Time (s)');
ylabel('Position');
title('Kalman Filter for Quadrotor Tracking');
legend;
grid on;
