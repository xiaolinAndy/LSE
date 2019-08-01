import numpy as np
import time
from utils import data_acquire
from scipy.stats import norm

class Algo:
    def __init__(self, D, GP_prior, h, imp, cost, beta=1.96, sigma_noise=np.exp(-1)):
        self.D = D
        self.mu = GP_prior[0]
        self.kernel_sigma = GP_prior[1]
        self.kernel_l = GP_prior[2]
        self.h = h
        self.beta = beta
        self.noise_sigma = sigma_noise
        self.imp = imp
        self.cost = cost
        self.query_point = np.zeros((D.shape[0], 2))
        self.query_value = np.zeros((D.shape[0]))
        self.query_num = 0

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        return None, None, None, None, None

    def select_point(self, D, C, last_point, flag_u, flag_z):
        return None, None

    def kernel(self, point_a, point_b):
        return self.kernel_sigma ** 2 * np.exp((-np.linalg.norm(point_a - point_b, axis=1) ** 2) / (2 * self.kernel_l ** 2))

    def get_K_inv(self, query_point):
        K = np.zeros((query_point.shape[0], query_point.shape[0]))
        for i in range(K.shape[0]):
            K[i, :] = self.kernel(query_point[i], query_point)
        K_inv = np.linalg.inv(K + self.noise_sigma ** 2 * np.eye(K.shape[0]))
        return K_inv

    # update parameters for GP, query_point (K, 2), query_value (K,)
    def update_GP(self, query_point, query_value, U, flag_u):
        K_inv = self.get_K_inv(query_point)
        mu = np.zeros((U.shape[0]))
        sigma = np.zeros((U.shape[0]))
        for i in range(U.shape[0]):
            if not flag_u[i]:
                continue
            kx = self.kernel(U[i, :], query_point)
            mu[i] = np.dot(np.dot(kx, K_inv), query_value)
            sigma[i] = self.kernel(U[i, :].reshape(1, 2), U[i, :].reshape(1, 2)) - np.dot(
                np.dot(kx, K_inv), kx)
        return mu, sigma

    # only update sigma for GP, query_point (K, 2), return k(x,x)
    def update_sigma(self, query_point, U, flag_u):
        K_inv = self.get_K_inv(query_point)
        sigma = np.zeros((U.shape[0]))
        for i in range(U.shape[0]):
            if not flag_u[i]:
                continue
            kx = self.kernel(U[i, :], query_point)
            sigma[i] = self.kernel(U[i, :].reshape(1, 2), U[i, :].reshape(1, 2)) - np.dot(
                np.dot(kx, K_inv), kx)
        return sigma

    # only update sigma for GP, query_point (K, 2), return k(x,x')
    def update_cov(self, query_point, U):
        K_inv = self.get_K_inv(query_point)
        cov = np.zeros((U.shape[0], U.shape[0]))
        for i in range(U.shape[0]):
            kx = self.kernel(U[i, :], query_point)
            cov[i, :] = self.kernel(U[i, :].reshape(1, 2), U) - np.dot(np.dot(kx, K_inv), kx)
        return cov

    # calculate F1 value
    def cal_F1(self, D, H, h, imp):
        y = data_acquire(D[:, 0], D[:, 1], [0, 0])
        # the total num of H
        num_t = (y > 1).sum()
        # calculate implicit threshold
        if imp:
            max_y = np.max(y)
            h = max_y * h
        TP, FP, FN = 0., 0., 0.
        for sample in H:
            y = data_acquire(sample[0], sample[1], [0, 0])
            if y > h:
                TP += 1
            else:
                FP += 1
        if TP + FP == 0 or TP == 0:
            return 0
        prec = TP / (TP + FP)
        recall = TP / num_t
        F1 = 2 * prec * recall / (prec + recall)
        return F1

    # core function
    def run(self, start_index):
        # some initialization
        H, L, D = [], [], self.D
        query_history = {}
        step = 1
        max_step = 100
        costs = 0
        F1, total_cost = [], []
        flag_u = np.ones(D.shape[0]) > 0
        flag_z = np.ones(D.shape[0]) > 0
        # C is confidential range, initialized as R
        C = np.zeros((D.shape[0], 2), dtype=float)
        C[:, 0] = -1e9 * np.ones(D.shape[0])
        C[:, 1] = 1e9 * np.ones(D.shape[0])
        # observe data
        y = data_acquire(D[start_index, 0], D[start_index, 1], [0, self.noise_sigma])
        self.query_point[self.query_num, :] = D[start_index, :]
        self.query_value[self.query_num] = y
        self.query_num += 1
        query_history[tuple(D[start_index, :])] = [y]
        last_point = D[start_index, :]
        # start timing
        s_time = time.time()
        while flag_u.any() and step <= max_step:
            C, H, L, flag_u, flag_z = self.estiamte(D, C, self.query_point, self.query_value, self.query_num, H, L, flag_u, flag_z)
            if not flag_u.any():
                break
            # select next point
            max_index, dist = self.select_point(D, C, last_point, flag_u, flag_z)
            costs += dist
            total_cost.append(costs)
            last_point = D[max_index, :]
            # observe point
            y = data_acquire(D[max_index, 0], D[max_index, 1], [0, self.noise_sigma])
            if tuple(D[max_index, :]) in query_history.keys():
                query_history[tuple(D[max_index, :])] += [y]
                for j in range(self.query_num - 1, -1):
                    if self.query_point[j, :] == D[max_index, :]:
                        tmp_sum = 0
                        for value in query_history[tuple(D[max_index, :])]:
                            tmp_sum += value
                        self.query_value[j] = tmp_sum / len(query_history[tuple(D[max_index, :])])
                        break
            else:
                self.query_point[self.query_num, :] = D[max_index, :]
                self.query_value[self.query_num] = y
                self.query_num += 1
            F1_score = self.cal_F1(D, H, self.h, self.imp)
            F1.append(F1_score)
            if step % 10 == 0:
                print('step: ', step, 'max_index: ', max_index, 'point: ', D[max_index, :], 'H: ', len(H), 'L: ', len(L),
                  'unclassified: ', D.shape[0] - len(H) - len(L), 'F1: ', F1_score, 'time: ', time.time() - s_time)
            step += 1
        points = self.query_point[0:20, :]
        self.query_point = np.zeros((D.shape[0], 2))
        self.query_value = np.zeros((D.shape[0]))
        self.query_num = 0
        return F1, total_cost, time.time() - s_time, points

class LSE(Algo):
    def __init__(self, D, GP_prior, h, imp, cost, acc=0):
        Algo.__init__(self, D, GP_prior, h, imp, cost)
        self.acc = acc
        self.name = 'LSE'

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        mu, sigma = self.update_GP(query_point[:query_num, :], query_value[:query_num], D, flag_u)
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            Q = np.zeros(2)
            Q[0] = mu[i] - self.beta ** 0.5 * sigma[i]
            Q[1] = mu[i] + self.beta ** 0.5 * sigma[i]
            # intersection of Q and C
            C[i, 0] = max(C[i, 0], Q[0])
            C[i, 1] = min(C[i, 1], Q[1])
            # classification
            if C[i, 0] + self.acc > self.h:
                H.append(D[i, :])
                flag_u[i] = False
            elif C[i, 1] - self.acc < self.h:
                L.append(D[i, :])
                flag_u[i] = False
        return C, H, L, flag_u, flag_z

    def select_point(self, D, C, last_point, flag_u, flag_z):
        A = np.ones((D.shape[0])) * -1e9
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            A[i] = min(C[i, 1] - self.h, self.h - C[i, 0])
            dist = np.sqrt((last_point[0] - D[i, 0]) ** 2 + (last_point[1] - D[i, 1]) ** 2) + 0.01
            if self.cost:
                A[i] /= dist
        max_index = np.argmax(A)
        dist = np.sqrt((last_point[0] - D[max_index, 0]) ** 2 + (last_point[1] - D[max_index, 1]) ** 2) + 0.01
        return max_index, dist

class LSE_imp(Algo):
    def __init__(self, D, GP_prior, h, imp, cost, acc=0):
        Algo.__init__(self, D, GP_prior, h, imp, cost)
        self.acc = acc
        self.name = 'LSE_imp'

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        mu, sigma = self.update_GP(query_point[:query_num, :], query_value[:query_num], D, flag_z)
        for i in range(D.shape[0]):
            if not flag_z[i]:
                continue
            Q = np.zeros(2)
            Q[0] = mu[i] - self.beta ** 0.5 * sigma[i]
            Q[1] = mu[i] + self.beta ** 0.5 * sigma[i]
            # intersection of Q and C
            C[i, 0] = max(C[i, 0], Q[0])
            C[i, 1] = min(C[i, 1], Q[1])
            # estimate threshold
            f_opt = np.max(C[flag_z, 1])
            f_pes = np.max(C[flag_z, 0])
            h_opt = self.h * f_opt
            h_pes = self.h * f_pes
            # classification
            if not flag_u[i]:
                continue
            if C[i, 0] + self.acc > h_opt:
                flag_u[i] = False
                H.append(D[i, :])
                if C[i, 1] < f_pes:
                    flag_z[i] = False
            elif C[i, 1] - self.acc < h_pes:
                flag_u[i] = False
                L.append(D[i, :])
                if C[i, 1] < f_pes:
                    flag_z[i] = False
        #print(h_opt, h_pes)
        return C, H, L, flag_u, flag_z

    def select_point(self, D, C, last_point, flag_u, flag_z):
        A = np.ones((D.shape[0])) * -1e9
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            A[i] = C[i, 1] - C[i, 0]
            dist = np.sqrt((last_point[0] - D[i, 0]) ** 2 + (last_point[1] - D[i, 1]) ** 2) + 0.01
            if self.cost:
                A[i] /= dist
        max_index = np.argmax(A)
        dist = np.sqrt((last_point[0] - D[max_index, 0]) ** 2 + (last_point[1] - D[max_index, 1]) ** 2) + 0.01
        return max_index, dist

class LSE_imp_mod(LSE_imp):
    def __init__(self, D, GP_prior, h, imp, cost, acc=0):
        LSE_imp.__init__(self, D, GP_prior, h, imp, cost, acc=0)
        self.name = 'LSE_imp_mod'

    def select_point(self, D, C, last_point, flag_u, flag_z):
        A = np.ones((D.shape[0])) * -1e9
        # re-estiamte
        f_opt = np.max(C[flag_z, 1])
        f_pes = np.max(C[flag_z, 0])
        h_opt = self.h * f_opt
        h_pes = self.h * f_pes
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            A[i]  = min(C[i, 1] - h_pes, h_opt - C[i, 0])
            dist = np.sqrt((last_point[0] - D[i, 0]) ** 2 + (last_point[1] - D[i, 1]) ** 2) + 0.01
            if self.cost:
                A[i] /= dist
        max_index = np.argmax(A)
        dist = np.sqrt((last_point[0] - D[max_index, 0]) ** 2 + (last_point[1] - D[max_index, 1]) ** 2) + 0.01
        return max_index, dist

class TRUVAR(Algo):
    def __init__(self, D, GP_prior, h, imp, cost, delta=0, eta=1, r=0.1):
        Algo.__init__(self, D, GP_prior, h, imp, cost)
        self.delta = delta
        self.eta = eta
        self.r = r
        self.name = 'TRUVAR'

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        mu, sigma = self.update_GP(query_point[:query_num, :], query_value[:query_num], D, flag_u)
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            Q = np.zeros(2)
            Q[0] = mu[i] - self.beta ** 0.5 * sigma[i]
            Q[1] = mu[i] + self.beta ** 0.5 * sigma[i]
            # classification
            if Q[0] > self.h:
                H.append(D[i, :])
                flag_u[i] = False
            elif Q[1] < self.h:
                L.append(D[i, :])
                flag_u[i] = False
        # update parameters
        if flag_u.any():
            max_sigma = np.max(self.beta ** 0.5 * sigma[flag_u])
            while max_sigma <= (1 + self.delta) * self.eta:
                self.eta *= self.r
        return None, H, L, flag_u, flag_z

    def select_point(self, D, C, last_point, flag_u, flag_z):
        A = np.zeros((D.shape[0]))
        trunc = self.eta ** 2
        sigma_old = self.update_sigma(self.query_point[:self.query_num, :], D, flag_u)
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            self.query_point[self.query_num, :] = D[i, :]
            sigma_new = self.update_sigma(self.query_point[:self.query_num + 1, :], D, flag_u)
            for j in range(D.shape[0]):
                if not flag_u[i]:
                    continue
                A[i] += max(self.beta ** 2 * sigma_old[j], trunc) - max(self.beta ** 2 * sigma_new[j], trunc)
            dist = np.sqrt((last_point[0] - D[i, 0]) ** 2 + (last_point[1] - D[i, 1]) ** 2) + 0.01
            if self.cost:
                A[i] /= dist
        max_index = np.argmax(A)
        dist = np.sqrt((last_point[0] - D[max_index, 0]) ** 2 + (last_point[1] - D[max_index, 1]) ** 2) + 0.01
        return max_index, dist

class TRUVAR_imp(TRUVAR):
    def __init__(self, D, GP_prior, h, imp, cost, delta=0, eta=1, r=0.1):
        TRUVAR.__init__(self, D, GP_prior, h, imp, cost, delta=0, eta=1, r=0.1)
        self.name = 'TRUVAR_imp'

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        mu, sigma = self.update_GP(query_point[:query_num, :], query_value[:query_num], D, flag_z)
        Q = np.zeros((D.shape[0], 2))
        Q[:, 0] = mu - self.beta ** 0.5 * sigma
        Q[:, 1] = mu + self.beta ** 0.5 * sigma
        f_opt, f_pes = np.max(Q[:, 1]), np.max(Q[:, 0])
        # estimate threshold
        h_opt = self.h * f_opt
        h_pes = self.h * f_pes
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            # classification
            if Q[i, 0] > h_opt:
                flag_u[i] = False
                H.append(D[i, :])
                if Q[i, 0] < f_pes:
                    flag_z[i] = False
            elif Q[i, 1] < h_pes:
                flag_u[i] = False
                L.append(D[i, :])
                if Q[i, 1] < f_pes:
                    flag_z[i] = False
        # update parameters
        if flag_u.any():
            max_sigma = np.max(self.beta ** 0.5 * sigma[flag_u])
            while max_sigma <= (1 + self.delta) * self.eta:
                self.eta *= self.r
        return None, H, L, flag_u, flag_z

class RMILE(Algo):
    def __init__(self, D, GP_prior, h, imp, cost, eta=0.01):
        Algo.__init__(self, D, GP_prior, h, imp, cost)
        self.eta = eta
        self.name = 'RMILE'

    def estiamte(self, D, C, query_point, query_value, query_num, H, L, flag_u, flag_z):
        mu, sigma = self.update_GP(query_point[:query_num, :], query_value[:query_num], D, flag_u)
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            Q = np.zeros(2)
            Q[0] = mu[i] - self.beta ** 0.5 * sigma[i]
            Q[1] = mu[i] + self.beta ** 0.5 * sigma[i]
            # classification
            if Q[0] > self.h:
                H.append(D[i, :])
                flag_u[i] = False
            elif Q[1] < self.h:
                L.append(D[i, :])
                flag_u[i] = False
        return None, H, L, flag_u, flag_z

    def select_point(self, D, C, last_point, flag_u, flag_z):
        A = np.zeros(D.shape[0])
        flag_all = A == 0
        mu_old, sigma_old = self.update_GP(self.query_point[:self.query_num, :], self.query_value[:self.query_num], D, flag_all)
        cov = self.update_cov(self.query_point[:self.query_num, :], D)
        for i in range(D.shape[0]):
            if not flag_u[i]:
                continue
            self.query_point[self.query_num, :] = D[i, :]
            sigma_new = np.array(sigma_old)
            tmp_div = sigma_old[i] ** 2 + self.noise_sigma ** 2
            for j in range(D.shape[0]):
                sigma_new[j] = sigma_old[j] - (cov[i][j] ** 2) / tmp_div
                den = tmp_div ** 0.5 * (mu_old[j] - self.beta * sigma_new[j] - self.h) / cov[i][j]
                A[i] += norm.cdf(den)
            I_gpeps = ((mu_old - self.beta * sigma_old) > self.h).sum()
            A[i] = max(A[i] - I_gpeps, self.eta * sigma_old[i])
            dist = np.sqrt((last_point[0] - D[i, 0]) ** 2 + (last_point[1] - D[i, 1]) ** 2) + 0.01
            if self.cost:
                A[i] /= dist
        max_index = np.argmax(A)
        dist = np.sqrt((last_point[0] - D[max_index, 0]) ** 2 + (last_point[1] - D[max_index, 1]) ** 2) + 0.01
        return max_index, dist