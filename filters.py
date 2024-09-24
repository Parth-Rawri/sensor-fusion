import numpy as np
import scipy


class KalmanFilter:
    def __init__(self, mu0, sigma0, Q, R, dt, sensor):
        self.A = np.vstack((np.hstack((np.eye(2), dt * np.eye(2))), np.hstack((np.zeros((2, 2)), np.eye(2)))))
        self.B = np.vstack((np.zeros((2, 2)), dt * np.eye(2)))
        self.C = np.hstack((np.eye(2), np.zeros((2, 2))))
        self.Q = np.block([[np.zeros((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), Q]])
        self.R = R
        self.dt = dt
        self.mu = mu0
        self.sigma = sigma0

    def predict(self, u):
        self.mu = self.A @ self.mu + self.B @ u
        self.sigma = self.A @ self.sigma @ self.A.T + self.Q 

    def update(self, y):
        self.K = self.sigma @ self.C.T @ np.linalg.inv(self.C @ self.sigma @ self.C.T + self.R)
        self.mu = self.mu + self.K @ (y - self.C @ self.mu)
        self.sigma = self.sigma - self.K @ self.C @ self.sigma

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.mu, self.sigma

    def plot_ellipse(self, P, mu, Sigma, ax, label="", color="green"):
        mu = mu.reshape((2, 1))
        r = np.sqrt(-2*np.log(1-P))
        theta = np.linspace(0, 2*np.pi)
        w = np.stack((r*np.cos(theta), r*np.sin(theta)))
        x = scipy.linalg.sqrtm(Sigma)@w + mu
        ax.plot(x[0,:], x[1,:], label=label, c=color)

    def plot_means_sigmas(self, P, mus, sigmas, ax):
        mus = np.array(mus)
        ax.plot(mus[:, 0], mus[:, 1], label="KF", color="red")
        for (mu, sigma) in zip(mus, sigmas):
            self.plot_ellipse(P, mu, sigma, ax)


class EKF:
    def __init__(self, mu0, sigma0, dt, base_station_locs):
        self.Q = 0.1 * dt * np.eye(3)
        self.R = 0.1 * np.eye(2)
        self.dt = dt
        self.mu = mu0
        self.sigma = sigma0
        self.base_station_locs = base_station_locs

    def f(self, u):
        x, y, theta = self.mu
        v, omega = u
        x_new = x + self.dt * v * np.cos(theta)
        y_new = y + self.dt * v * np.sin(theta)
        theta_new = theta + self.dt * omega
        return np.array([x_new, y_new, theta_new])

    def g(self):
        p = self.mu[0:2]
        m = np.zeros(2)
        m[0] = np.linalg.norm(p - self.base_station_locs[0])
        m[1] = np.linalg.norm(p - self.base_station_locs[1])
        return m

    def dynamicsJacob(self, u):
        A = np.eye(3)
        v, theta = u[0], self.mu[2]
        A[0, 2] = - self.dt * v * np.sin(theta)
        A[1, 2] = self.dt * v * np.cos(theta)
        return A

    def measurementJacob(self):
        p = self.mu[0:2]  # Extract position [Px, Py]
        C = np.zeros((2, 3))  # 2 measurements (rows), 3 state variables (columns)
        
        # Range to base station 1 (first measurement row)
        delta_p1 = p - self.base_station_locs[0]
        norm_p1 = np.linalg.norm(delta_p1)
        C[0, 0] = delta_p1[0] / norm_p1  # Partial derivative wrt Px
        C[0, 1] = delta_p1[1] / norm_p1  # Partial derivative wrt Py

        # Range to base station 2 (second measurement row)
        delta_p2 = p - self.base_station_locs[1]
        norm_p2 = np.linalg.norm(delta_p2)
        C[1, 0] = delta_p2[0] / norm_p2  # Partial derivative wrt Px
        C[1, 1] = delta_p2[1] / norm_p2  # Partial derivative wrt Py
        return C

    def predict(self, u):
        self.mu = self.f(u)
        A = self.dynamicsJacob(u)
        self.sigma = A @ self.sigma @ A.T + self.Q   

    def update(self, y):
        C = self.measurementJacob()
        K = self.sigma @ C.T @ np.linalg.inv(C @ self.sigma @ C.T + self.R)
        self.mu = self.mu + K @ (y - self.g())
        self.sigma = self.sigma - K @ C @ self.sigma

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.mu, self.sigma

    def plot_ellipse(self, P, mu, Sigma, ax, label="", color="green"):
        mu = mu.reshape((2, 1))
        r = np.sqrt(-2*np.log(1-P))
        theta = np.linspace(0, 2*np.pi)
        w = np.stack((r*np.cos(theta), r*np.sin(theta)))
        x = scipy.linalg.sqrtm(Sigma)@w + mu
        ax.plot(x[0,:], x[1,:], label=label, c=color)

    def plot_means_sigmas(self, P, mus, sigmas, ax):
        mus = np.array(mus)
        ax.plot(mus[:, 0], mus[:, 1], label="EKF", color="red")
        for (mu, sigma) in zip(mus, sigmas):
            self.plot_ellipse(P, mu, sigma, ax)


class UKF:
    def __init__(self, mu0, sigma0, dt, base_station_locs):
        self.Q = 0.1 * dt * np.eye(3)
        self.R = 0.1 * np.eye(2)
        self.dt = dt
        self.mu = mu0
        self.sigma = sigma0
        self.base_station_locs = base_station_locs
        self.lam = 2

    def UT(self, mu, sigma):
        n = mu.shape[0]
        x0, w0 = mu, self.lam/(self.lam + n)
        sigma_pts = [(x0,w0)]
        M = scipy.linalg.sqrtm((self.lam + n) * sigma)
        for idx in range(n):
            x1, x2 = mu + M[:, idx], mu - M[:, idx]
            w = 0.5 / (self.lam + n)
            sigma_pts.append((x1, w))
            sigma_pts.append((x2, w))
        return sigma_pts
    
    def UTinv(self, sigma_pts, cov):
        n = sigma_pts[0][0].shape[0]
        mu, sigma = np.zeros(n), np.zeros((n, n))
        for (xi, wi) in sigma_pts:
            mu += wi * xi
        for (xi, wi) in sigma_pts:
            sigma += wi * ((xi - mu).reshape(-1,1) @ (xi - mu).reshape(1,-1))
        return mu, sigma + cov

    def f(self, xi, u):
        x, y, theta = xi
        v, omega = u
        x_new = x + self.dt * v * np.cos(theta)
        y_new = y + self.dt * v * np.sin(theta)
        theta_new = theta + self.dt * omega
        return np.array([x_new, y_new, theta_new])

    def g(self, xi):
        p = xi[0:2]
        m = np.zeros(2)
        m[0] = np.linalg.norm(p - self.base_station_locs[0])
        m[1] = np.linalg.norm(p - self.base_station_locs[1])
        return m

    def predict(self, u):
        sigma_pts = self.UT(self.mu, self.sigma)
        pred_sigma_pts = []
        for (xi, wi) in sigma_pts:
            xi_pred = self.f(xi, u)
            pred_sigma_pts.append((xi_pred, wi))
        self.mu, self.sigma = self.UTinv(pred_sigma_pts, self.Q)

    def update(self, y):
        sigma_pts = self.UT(self.mu, self.sigma)
        pred_meas_sigma_pts = []
        for (xi, wi) in sigma_pts:
            yi = self.g(xi)
            pred_meas_sigma_pts.append((yi, wi))
        
        mu_y, sigma_y = self.UTinv(pred_meas_sigma_pts, self.R)
        sigma_xy = np.zeros((3, 2))
        for i in range(len(sigma_pts)):
            xi, wi = sigma_pts[i]
            yi, _ = pred_meas_sigma_pts[i]
            sigma_xy += wi * (xi - self.mu).reshape(-1, 1) @ (yi - mu_y).reshape(1, -1)

        self.mu += sigma_xy @ np.linalg.inv(sigma_y) @ (y - mu_y)
        self.sigma -= sigma_xy @ np.linalg.inv(sigma_y) @ sigma_xy.T

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.mu, self.sigma

    def plot_ellipse(self, P, mu, Sigma, ax, label="", color="green"):
        mu = mu.reshape((2, 1))
        r = np.sqrt(-2*np.log(1-P))
        theta = np.linspace(0, 2*np.pi)
        w = np.stack((r*np.cos(theta), r*np.sin(theta)))
        x = scipy.linalg.sqrtm(Sigma)@w + mu
        # ax.plot(x[0,:], x[1,:], label=label, c=color)

    def plot_means_sigmas(self, P, mus, sigmas, ax):
        mus = np.array(mus)
        ax.plot(mus[:, 0], mus[:, 1], label="UKF", color="red")
        for (mu, sigma) in zip(mus, sigmas):
            self.plot_ellipse(P, mu, sigma, ax)


class PF:
    def __init__(self, mu0, sigma0, dt, base_station_locs, num_particles):
        self.Q = 0.1 * dt * np.eye(3)
        self.R = 0.1 * np.eye(2)
        self.dt = dt
        self.base_station_locs = base_station_locs
        self.num_particles = num_particles
        self.x = np.random.multivariate_normal(mean=mu0, cov=sigma0, size=num_particles)
        self.w = np.array([1.0/self.num_particles] * self.num_particles)

    def f(self, xi, u):
        x, y, theta = xi
        v, omega = u
        x_new = x + self.dt * v * np.cos(theta)
        y_new = y + self.dt * v * np.sin(theta)
        theta_new = theta + self.dt * omega
        return np.array([x_new, y_new, theta_new])
    
    def g(self, xi):
        p = xi[0:2]
        m = np.zeros(2)
        m[0] = np.linalg.norm(p - self.base_station_locs[0])
        m[1] = np.linalg.norm(p - self.base_station_locs[1])
        return m
    
    def resample(self):
        inds = np.random.choice(self.num_particles, size=self.num_particles, p=self.w)
        self.x = self.x[inds]
        self.w = np.array([1.0 / self.num_particles] * self.num_particles)

    def predict(self, u):
        for i in range(self.num_particles):
            self.x[i] = self.f(self.x[i], u) + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Q)

    def update(self, y):
        R_inv = np.linalg.inv(self.R)
        w = np.zeros_like(self.w)
        for i in range(self.num_particles):
            w[i] = np.exp(-0.5 * (y - self.g(self.x[i])).T @ R_inv @ (y - self.g(self.x[i])))
        self.w = w / np.sum(w)

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        self.resample()
        mu, sigma = np.zeros(3), np.zeros((3, 3))
        for i in range(self.num_particles):
            mu += self.w[i] * self.x[i]
        for i in range(self.num_particles):
            sigma += self.w[i] * np.outer(self.x[i] - mu, self.x[i] - mu)
        return mu, sigma

    def plot_ellipse(self, P, mu, Sigma, ax, label="", color="green"):
        mu = mu.reshape((2, 1))
        r = np.sqrt(-2*np.log(1-P))
        theta = np.linspace(0, 2*np.pi)
        w = np.stack((r*np.cos(theta), r*np.sin(theta)))
        x = scipy.linalg.sqrtm(Sigma)@w + mu
        # ax.plot(x[0,:], x[1,:], label=label, c=color)

    def plot_means_sigmas(self, P, mus, sigmas, ax):
        mus = np.array(mus)
        ax.plot(mus[:, 0], mus[:, 1], label="UKF", color="red")
        for (mu, sigma) in zip(mus, sigmas):
            self.plot_ellipse(P, mu, sigma, ax)
