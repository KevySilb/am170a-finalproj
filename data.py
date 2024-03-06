import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as itg
from matplotlib import gridspec
import copy

class simulateBacteria:
    def __init__(self, init:tuple, t_points:np.arange, model:str = 'harvest') -> None:
        self.x0, self.y0 = init
        self.t_points = t_points

        self.Insulin = 0

        self.cumsumInsulin = []
        self.lambda_values = []
        if model == 'extract':
            self.sol = self.simulate(harvest=False)
        elif model == 'harvest':
            self.sol = self.simulate(harvest=True)

    def lambda_func(self, I):
        return 4 * (np.tanh(1e2 * I))**2
    
    def func(self, t, r):
        """
        x1 represents millions of bacteria in solution at a particular point in time 
        x2 represents mg of insulin per mL
        """
        x1, x2 = r
        lambda_val = self.lambda_func(x2)
        self.lambda_values.append(lambda_val)

        extractedInsulin = lambda_val * x2
        # running total of extracted insulin
        self.Insulin += extractedInsulin
        # append new total to list for plotting 
        self.cumsumInsulin.append(self.Insulin)
        
        # the ivp:
        dx1_dt = 0.0355 * x1 * (1-(x1/1e4)) - 1.68 * x1 * x2 
        dx2_dt = 5.41e-08 * x1 - extractedInsulin
        return dx1_dt, dx2_dt

    def func2(self, t, r):
        """
        x1 represents millions of bacteria in solution at a particular point in time 
        x2 represents mg of insulin per mL
        """
        x1, x2 = r

        # the ivp:
        dx1_dt = 0.0355 * x1 * (1-(x1/1e4)) - 1.68 * x1 * x2 
        dx2_dt = 5.41e-08 * x1
        return dx1_dt, dx2_dt
        
    def simulate(self, harvest = False):
        # time interval
        t0 = self.t_points[0]
        tf = self.t_points[-1]
        
        # initial conditions
        init = (self.x0, self.y0)
        if harvest:
            result = itg.solve_ivp(self.func2, [t0, tf], init, t_eval=self.t_points)
        else:
            result = itg.solve_ivp(self.func, [t0, tf], init, t_eval=self.t_points)
        return result

def hyperbolic_tan(x):
    amplitude = 1
    a = 1e2
    y = x * a
    return amplitude * (np.tanh(y))**2

def figure_results():
    init = (100, 0)
    t0 = 0
    tf = 600
    t_eval = np.arange(t0, tf, 1)
    fig, ax = plt.subplots()
    sim1 = simulateBacteria(init, t_eval, 'extract')
    sim2 = simulateBacteria(init, t_eval, 'harvest')
    # ax.plot(np.linspace(t0, tf, (len(sim1.lambda_values))), sim1.lambda_values, label = 'lambda')
    bac_e, insul_e = sim1.sol.y
    bac_h, insul_h = sim2.sol.y
    ax.plot(t_eval, insul_h, label = 'insulin harvest')
    ax.plot(np.linspace(t0, tf, len(sim1.cumsumInsulin)), sim1.cumsumInsulin, label = 'extracted insulin')
    # ax.set_yscale('log')
    ax.legend(loc = 'best')
    plt.show()

def plot_lambda():
    data = pd.read_csv('data_with_insulin.txt')
    x = np.linspace(0, data['i3'].values[-1], 100)
    fig, ax = plt.subplots()
    ax.plot(x, hyperbolic_tan(x), label = r'$\lambda(I)$')
    ax.axvline(data['i3'].values[-1], c = 'r', alpha = 0.4, linestyle = '--', label = r'mg Insulin harvested at 600 minutes')
    ax.set_title('Lambda function')
    ax.set_xlabel('Insulin Concentration (mg/mL)')
    ax.set_ylabel(r'extraction constant $\lambda(I)$')
    ax.legend(loc = 'upper left')
    plt.show()

def tmax():
    init = (100, 0)
    t0 = 0
    tf = 600
    t_eval = np.arange(t0, tf, 1)
    sim = simulateBacteria(init, t_eval, 'harvest')
    bac, insulin = sim.sol.y
    dI_dt = np.gradient(insulin, t_eval)
    d2I_dt2 = np.gradient(dI_dt, t_eval)
    max_dI_dt = np.argmax(dI_dt)
    optimal_harvest_time = t_eval[max_dI_dt]

    # Creating a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting on the first subplot
    axs[0].plot(t_eval, insulin)
    axs[0].set_title('Insulin over time')
    axs[0].set_xlabel('minutes')
    axs[0].set_ylabel('Insulin')

    # Plotting on the second subplot
    axs[1].plot(t_eval, dI_dt, label=r'$\frac{dI}{dt}$')
    axs[1].set_title('Rate of change of Insulin')
    axs[1].set_xlabel('minutes')
    axs[1].set_ylabel(r'$\frac{dI}{dt}$')
    axs[1].legend()

    # Plotting on the third subplot
    axs[2].plot(t_eval, d2I_dt2, c='r')
    axs[2].set_title('Second derivative of Insulin over time')
    axs[2].set_xlabel('minutes')
    axs[2].set_ylabel(r'$\frac{d^2I}{dt^2}$')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def result1():
    batches = 4
    init = (100, 0)
    t0 = 0
    t_max = 153
    tf = t_max * batches
    t_eval = np.arange(t0, tf+1, 1)
    sim = simulateBacteria(init, t_eval, 'harvest')
    simE = simulateBacteria(init, t_eval, 'extract')
    bac, insulin = sim.sol.y
    t_interval = np.arange(0, t_max*(batches+1), t_max)

    # generate step data of harvesting every 153 minutes
    insul_harvest = 0
    cumsumI_harvest = []
    for i in range(len(t_interval)):
        cumsumI_harvest.append(insul_harvest)
        insul_harvest += insulin[-1]
    plt.step(t_interval, cumsumI_harvest, where = 'post', label = 'batch every 153 minutes')
    plt.plot(np.linspace(t0, tf, len(simE.cumsumInsulin)), simE.cumsumInsulin, label = 'extracted insulin')
    plt.xlabel('minutes')
    plt.ylabel('mg Insulin')
    plt.title('Batch harvesting vs continuous extraction')
    plt.legend(loc = 'upper left')
    plt.show()

def main():
    tmax()




if __name__ == '__main__':
    main()
    