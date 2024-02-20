import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as itg
from scipy.optimize import minimize
'''1mg/hour per mL for a bacterial solution containing 0.1g of dry bacteria per mL'''

class simulateBacteria:
    
    def __init__(self, kappa, init, t_points) -> None:
        self.a, self.b, self.g = 0.035, 1.57, 5.41e-08
        self.k = kappa
        self.x0, self.y0 = init
        self.t_points = t_points
        self.sol = self.simulate()
    
    def func(self, t, r):
        x, y = r
        dx_dt = self.a*x*(1-(x/self.k)) - self.b*x*y 
        dy_dt = self.g*x
        return dx_dt, dy_dt
    
    def simulate(self):
        # time interval
        t0 = self.t_points[0]
        tf = self.t_points[-1]
        
        # initial conditions
        init = (self.x0, self.y0)
        
        result = itg.solve_ivp(self.func, [t0, tf], init, t_eval=self.t_points)
        return result
    
def objective_function(params, *args):
    data, t_eval, init = args
    sim = simulateBacteria(params, init, t_eval)
    error = np.sum(np.log(((data['y2'].values - sim.sol.y[0]) + 1e-8)**2) + np.log((((data['i2'].values - sim.sol.y[1]) + 1e-8))**2))
    return error

def main():
    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    data = pd.read_csv('data_with_insulin.txt')
    # params = (0.027, 1.07, 1000, 0.00000006)
    # params = (0.027, 1.518, 1000, 5.413e-08)
    k1, k2, k3 = 1e2, 1e3, 1e4
    init = (100, 0)
    sim1 = simulateBacteria(k1, init, data['t'].values)
    sim2 = simulateBacteria(k2, init, data['t'].values)
    sim3 = simulateBacteria(k3, init, data['t'].values)
    # bounds = [(1e-6, None), (1e-6, None), (1, None), (1e-10, None)]
    # result = minimize(objective_function, params, args = (data, data['t'].values, init), bounds=bounds)
    # alpha = bacteria growth rate
    # beta = insulins effect on bacteria
    # kappa = carrying capacity 
    # gamma = insulin production rate per bacteria
    # alpha, beta, kappa, gamma = result.x
    # print('alpha = {}, beta = {}, kappa = {}, gamma = {}'.format(alpha, beta, kappa, gamma))
    # sim_optimize = simulateBacteria(result.x, init, data['t'].values)
    # bacteria2, insulin2 = sim_optimize.sol.y
    bac1, insul1 = sim1.sol.y
    bac2, insul2 = sim2.sol.y
    bac3, insul3 = sim3.sol.y

    # Plotting model predictions
    ax.plot(data['t'].values, bac1, alpha=0.5, linestyle='-', c='#377eb8', label=r'$M_{\text{bacteria, sim1}}$')
    ax.plot(data['t'].values, insul1, alpha=0.5, linestyle='-', c='#ff7f00', label=r'$M_{\text{insulin, sim1}}$')
    ax.plot(data['t'].values, bac2, alpha=0.5, linestyle='-', c='#56B4E9', label=r'$M_{\text{bacteria, sim2}}$')
    ax.plot(data['t'].values, insul2, alpha=0.5, linestyle='-', c='#009E73', label=r'$M_{\text{insulin, sim2}}$')
    ax.plot(data['t'].values, bac3, alpha=0.5, linestyle='-', c='#D55E00', label=r'$M_{\text{bacteria, sim3}}$')
    ax.plot(data['t'].values, insul3, alpha=0.5, linestyle='-', c='#F0E442', label=r'$M_{\text{insulin, sim3}}$')

    # Plotting data points with error bars
    ax.errorbar(data['t'], data['y1'], data['err_y1'], alpha=0.5, linestyle='--', c='#377eb8', label=r'$D_{\text{bacteria, sim1}}$')
    ax.errorbar(data['t'], data['i1'], data['err_i1'], alpha=0.5, linestyle='--', c='#ff7f00', label=r'$D_{\text{insulin, sim1}}$')
    ax.errorbar(data['t'], data['y2'], data['err_y2'], alpha=0.5, linestyle='--', c='#56B4E9', label=r'$D_{\text{bacteria, sim2}}$')
    ax.errorbar(data['t'], data['i2'], data['err_i2'], alpha=0.5, linestyle='--', c='#009E73', label=r'$D_{\text{insulin, sim2}}$')
    ax.errorbar(data['t'], data['y3'], data['err_y3'], alpha=0.5, linestyle='--', c='#D55E00', label=r'$D_{\text{bacteria, sim3}}$')
    ax.errorbar(data['t'], data['i3'], data['err_i3'], alpha=0.5, linestyle='--', c='#F0E442', label=r'$D_{\text{insulin, sim3}}$')

    ax.set_yscale('log')
    ax.set_xlabel('minutes')
    ax.set_ylabel(r'$N(t), I(t)$')
    ax.set_title('Logarithmic Model vs Data')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
    