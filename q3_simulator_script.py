###############
# Script to work with the simulator for the mobile robot system in 
# HW6 Question 3.
###############

import numpy as np

import q3_simulator_class as sim
from filters import EKF, UKF, PF
import matplotlib.pyplot as plt
import time

# Define the parameters
n_steps = 100
pose0 = [0, 0, 0]

### EKF
mu0 = np.array([0, 0, 0])
sigma0 = 0.01 * np.eye(3)

robot = sim.MobileRobotSimulator()
phist, yhist, uhist = robot.simulate(mu0, n_steps)

p_mus = [mu0[0:2]]
p_sigmas = [sigma0[0:2, 0:2]]

ekf = EKF(mu0, sigma0, dt=0.5, base_station_locs=np.array([[30, -5], [-10, 20]]))

start_time = time.time()
for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    mu, sigma = ekf.step(u, y)

    p_mus.append(mu[0:2])
    p_sigmas.append(sigma[0:2, 0:2])

end_time = time.time()
time_taken = end_time - start_time
print(f"Average time for EKF is: {time_taken / len(phist)}")
fig_ekf = robot.plot_pose_history(phist, show_plot=False)
ekf.plot_means_sigmas(0.95, p_mus, p_sigmas, fig_ekf.gca())
plt.legend()
plt.show()


### UKF
mu0 = np.array([0, 0, 0])
sigma0 = 0.01 * np.eye(3)

robot = sim.MobileRobotSimulator()
phist, yhist, uhist = robot.simulate(mu0, n_steps)

p_mus = [mu0[0:2]]
p_sigmas = [sigma0[0:2, 0:2]]

ukf = UKF(mu0, sigma0, dt=0.5, base_station_locs=np.array([[30, -5], [-10, 20]]))

start_time = time.time()
for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    mu, sigma = ukf.step(u, y)

    p_mus.append(mu[0:2])
    p_sigmas.append(sigma[0:2, 0:2])

end_time = time.time()
time_taken = end_time - start_time
print(f"Average time for UKF is: {time_taken / len(phist)}")
fig_ukf = robot.plot_pose_history(phist, show_plot=False)
ukf.plot_means_sigmas(0.95, p_mus, p_sigmas, fig_ukf.gca())
plt.legend()
plt.show()


## PF
mu0 = np.array([0, 0, 0])
sigma0 = 0.01 * np.eye(3)

robot = sim.MobileRobotSimulator()
phist, yhist, uhist = robot.simulate(mu0, n_steps)

p_mus = [mu0[0:2]]
p_sigmas = [sigma0[0:2, 0:2]]

pf = PF(mu0, sigma0, dt=0.5, base_station_locs=np.array([[30, -5], [-10, 20]]), num_particles=1000)

start_time = time.time()
for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    mu, sigma = pf.step(u, y)

    p_mus.append(mu[0:2])
    p_sigmas.append(sigma[0:2, 0:2])

end_time = time.time()
time_taken = end_time - start_time
print(f"Average time for PF is: {time_taken / len(phist)}")
fig_pf = robot.plot_pose_history(phist, show_plot=False)
pf.plot_means_sigmas(0.95, p_mus, p_sigmas, fig_pf.gca())
plt.legend()
plt.show()