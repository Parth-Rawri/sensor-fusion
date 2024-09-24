###############
# Script to work with the simulator for the quadrotor system in 
# HW6 Question 1 and Question 2.
###############

import numpy as np
from filters import KalmanFilter
import matplotlib.pyplot as plt
import q1q2_simulator_class as sim

# Define the parameters
n_steps = 100
p0 = [1000, 0]
s0 = [0, 50]

### Question 1
quadrotor = sim.QuadrotorSimulator(sensor="GPS")
phist, shist, yhist, uhist = quadrotor.simulate(p0, s0, n_steps)

fig_pos = quadrotor.plot_position_history(phist, yhist, show_plot=True)
fig_pos.savefig("position_history.png", dpi=300, bbox_inches="tight")
fig_vel = quadrotor.plot_velocity_history(shist, show_plot=True)
fig_vel.savefig("velocity_history.png", dpi=300, bbox_inches="tight")
plt.show()

prun_hists, srun_hists, yrun_hists, urun_hists = quadrotor.simulate_multiple_runs(p0, s0, n_steps, 10)
quadrotor.plot_position_histories(prun_hists, yrun_hists, show_plot=True)
quadrotor.plot_velocity_histories(srun_hists, show_plot=True)

# Kalman Filter implementation
mu0 = np.array([1500, 100, 0, 55])
sigma0 = np.eye(4)
sigma0[0,0] = 250000
sigma0[1,1] = 250000
p0 = mu0[:2]
s0 = mu0[2:]

quadrotor = sim.QuadrotorSimulator(sensor="GPS")
phist, shist, yhist, uhist = quadrotor.simulate(p0, s0, n_steps)

p_mus = [mu0[0:2]]
p_sigmas = [sigma0[0:2, 0:2]]
s_mus = [mu0[2:]]
s_sigmas = [sigma0[2:, 2:]]
kalman = KalmanFilter(mu0, sigma0, np.eye(2), 9*np.eye(2), 1.0, "GPS")

for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    mu, sigma = kalman.step(u, y)

    p_mus.append(mu[0:2])
    p_sigmas.append(sigma[0:2, 0:2])
    
    s_mus.append(mu[2:])
    s_sigmas.append(sigma[2:, 2:])

fig_kf = quadrotor.plot_position_history(phist, yhist, show_plot=False)
kalman.plot_means_sigmas(0.95, p_mus, p_sigmas, fig_kf.gca())
plt.legend()
plt.show()

fig_kf = quadrotor.plot_velocity_history(shist, yhist, show_plot=False)
kalman.plot_means_sigmas(0.95, s_mus, s_sigmas, fig_kf.gca())
plt.legend()
plt.show()

### Question 2
mu0 = np.array([1000, 0, 0, 50])
sigma0 = np.eye(4)
p0 = mu0[:2]
s0 = mu0[2:]

quadrotor = sim.QuadrotorSimulator(sensor="Velocity")
phist, shist, yhist, uhist = quadrotor.simulate(p0, s0, n_steps)

p_mus = [mu0[0:2]]
p_sigmas = [sigma0[0:2, 0:2]]
s_mus = [mu0[2:]]
s_sigmas = [sigma0[2:, 2:]]
kalman = KalmanFilter(mu0, sigma0, np.eye(2), 9*np.eye(2), 1.0, "Velocity")

for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    mu, sigma = kalman.step(u, y)

    p_mus.append(mu[0:2])
    p_sigmas.append(sigma[0:2, 0:2])
    
    s_mus.append(mu[2:])
    s_sigmas.append(sigma[2:, 2:])

fig_kf = quadrotor.plot_position_history(phist, yhist, show_plot=False)
kalman.plot_means_sigmas(0.95, p_mus, p_sigmas, fig_kf.gca())
plt.legend()
plt.show()

fig_kf = quadrotor.plot_velocity_history(shist, yhist, show_plot=False)
kalman.plot_means_sigmas(0.95, s_mus, s_sigmas, fig_kf.gca())
plt.legend()
plt.show()