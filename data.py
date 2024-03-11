import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as itg
from matplotlib import gridspec
import copy

class simulateBacteria:
    def __init__(self, A:int, init:tuple, t_points:np.arange, model:str = 'harvest') -> None:
        self.x0, self.y0 = init
        self.t_points = t_points
        self.A = A
        self.kappa = 1e4
        self.Insulin = 0

        self.cumsumInsulin = []
        self.lambda_values = []
        if model == 'extract':
            self.sol = self.simulate(harvest=False)
        elif model == 'harvest':
            self.sol = self.simulate(harvest=True)

    def lambda_func(self, I):
        # return 4*33*I
        return self.A * (np.tanh(1e2 * I))**2
    
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
        dx1_dt = 0.0355 * x1 * (1-(x1/1e4)) - 1.77 * x1 * x2 
        dx2_dt = 5.2e-08 * x1 - extractedInsulin
        return dx1_dt, dx2_dt

    def func2(self, t, r):
        """
        x1 represents millions of bacteria in solution at a particular point in time 
        x2 represents mg of insulin per mL
        """
        x1, x2 = r

        # the ivp:
        dx1_dt = 0.0355 * x1 * (1-(x1/self.kappa)) - 1.77 * x1 * x2 
        dx2_dt = 5.2e-08 * x1
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
    ax.set_title(r'Lambda function $\lambda(I) = A \cdot \left(\tanh{\frac{I}{10^{-2}}}\right)^2$ where $A = 1$')
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
    optimal_harvest_time = t_eval[np.argmin(d2I_dt2)]

    # Creating a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plotting on the first subplot
    axs[0].plot(t_eval, insulin, label = 'I(t)')
    axs[0].set_title('Insulin over time in a single batch')
    axs[0].set_xlabel('minutes')
    axs[0].set_ylabel(r'mg Insulin per mL')

    # Plotting on the second subplot
    axs[1].plot(t_eval, dI_dt, label=r'$\frac{dI(t)}{dt}$')
    axs[1].set_title('First derivative of Insulin over time')
    axs[1].set_xlabel('minutes')
    axs[1].set_ylabel(r'$\frac{dI}{dt}$', rotation = 90)
    axs[1].legend()

    # Plotting on the third subplot
    axs[2].plot(t_eval, d2I_dt2, label = r'$\frac{d^2I(t)}{dt^2}$')
    axs[2].set_title('Second derivative of Insulin over time')
    axs[2].axvline(optimal_harvest_time, color='r', alpha = 0.5, linestyle='--', label = 't = {}'.format(optimal_harvest_time))
    axs[2].set_xlabel('minutes')
    axs[2].set_ylabel(r'$\frac{d^2I}{dt^2}$', rotation = 90)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def result1():
    params = [0.22, 0.28, 0.34]
    batches = 30
    init = (100, 0)
    t0 = 0
    t_max = 203
    tf = t_max * batches
    t_eval = np.arange(t0, tf+1, 1)
    sim = simulateBacteria(0, init, t_eval, 'harvest')
    _ , insulin = sim.sol.y
    t_interval = np.arange(t0, t_max*(batches+1), t_max)
    insul_harvest = 0
    cumsumI_harvest = []
    for i in range(len(t_interval)):
        cumsumI_harvest.append(insul_harvest)
        insul_harvest += insulin[-1]
    for param in params:
        simE = simulateBacteria(param, init, t_eval, 'extract')
        plt.plot(np.linspace(t0, tf, len(simE.cumsumInsulin)), simE.cumsumInsulin, label = 'A = {:1g}'.format(param))
        print((simE.cumsumInsulin[-1] - cumsumI_harvest[-1]) / cumsumI_harvest[-1])
    plt.step(t_interval, cumsumI_harvest, where = 'post', label = r'$\lambda(I) = 0$')
    plt.xlabel('minutes')
    plt.ylabel('mg Insulin/mL')
    plt.title(r'$\lambda(I) = A \cdot \left(\tanh{\frac{I}{10^{-2}}}\right)^2$ vs $\lambda(I) = 0$')
    plt.legend(loc = 'upper left', bbox_to_anchor = (1,1))
    plt.subplots_adjust(right = 0.8)
    plt.show()

def lambda_func_sim():

    init = (100, 0)
    t0 = 0
    tf = 400
    t_eval = np.arange(t0, tf+1, 1)
    simE = simulateBacteria(1, init, t_eval, 'extract')
    plt.plot(np.linspace(t0, tf, len(simE.lambda_values)), simE.lambda_values, label = 'lambda values')
    plt.xlabel('minutes')
    plt.ylabel('lambda')
    plt.title('lambda function values')
    plt.legend(loc = 'upper left')
    plt.show()

def plot_fit_model():
    data = pd.read_csv('data_with_insulin.txt')
    init = (100, 0)
    t0 = 0
    tf = int(data['t'].shape[0])*10
    t_eval = np.arange(t0, tf, 1)
    sim1 = simulateBacteria(1e2, init, t_eval, 'harvest')
    sim2 = simulateBacteria(1e3, init, t_eval, 'harvest')
    sim3 = simulateBacteria(1e4, init, t_eval, 'harvest')
    bac1, insul1 = sim1.sol.y
    bac2, insul2 = sim2.sol.y
    bac3, insul3 = sim3.sol.y
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot with bacteria and insulin
    axs[0].errorbar(data['t'].values, data['y1'].values, data['err_y1'].values, linewidth=4, alpha=0.5, label='Bacteria 1')
    axs[0].errorbar(data['t'].values, data['y2'].values, data['err_y2'].values, linewidth=4, alpha=0.5, label='Bacteria 2')
    axs[0].errorbar(data['t'].values, data['y3'].values, data['err_y3'].values, linewidth=4, alpha=0.5, label='Bacteria 3')
    axs[0].errorbar(data['t'].values, data['i1'].values, data['err_i1'].values, linewidth=4, alpha=0.5, label='Insulin 1')
    axs[0].errorbar(data['t'].values, data['i2'].values, data['err_i2'].values, linewidth=4, alpha=0.5, label='Insulin 2')
    axs[0].errorbar(data['t'].values, data['i3'].values, data['err_i3'].values, linewidth=4, alpha=0.5, label='Insulin 3')

    axs[0].plot(t_eval, bac1, color='red', linewidth=1.5)
    axs[0].plot(t_eval, bac2, color='red', linewidth=1.5)
    axs[0].plot(t_eval, bac3, color='red', linewidth=1.5)
    axs[0].plot(t_eval, insul1, color='red', linewidth=1.5)
    axs[0].plot(t_eval, insul2, color='red', linewidth=1.5)
    axs[0].plot(t_eval, insul3, color='red', linewidth=1.5)

    axs[0].set_xlabel('minutes')
    axs[0].set_ylabel('Log(mg/mL Insulin, million bacteria)')
    axs[0].set_yscale('log')
    axs[0].set_title('Logistic Model over Experimental data')


    # Right plot with only insulin
    axs[1].errorbar(data['t'].values, data['i1'].values, data['err_i1'].values, linewidth=4, alpha=0.5, label='Insulin 1')
    axs[1].errorbar(data['t'].values, data['i2'].values, data['err_i2'].values, linewidth=4, alpha=0.5, label='Insulin 2')
    axs[1].errorbar(data['t'].values, data['i3'].values, data['err_i3'].values, linewidth=4, alpha=0.5, label='Insulin 3')

    axs[1].plot(t_eval, insul1, color='red', linewidth=1.5, label='Logistic Model')
    axs[1].plot(t_eval, insul2, color='red', linewidth=1.5)
    axs[1].plot(t_eval, insul3, color='red', linewidth=1.5)

    axs[1].set_xlabel('minutes')
    axs[1].set_ylabel('mg/mL Insulin')
    axs[1].set_title('Logistic Model over Experimental data (Insulin Only)')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[0].legend(loc='lower right')
    plt.subplots_adjust(left=0.1, right= 0.85)
    plt.show()

def optimal_t():
    batches = 50
    init = (100, 0)
    delay = 5
    timeI = dict()
    t_values = []  # List to store t values for plotting
    insulin_values = []  # List to store corresponding insulin concentrations for plotting

    for i in np.arange(150, 300, 1):
        t_eval = np.linspace(0, i + 1, i + 1)
        sim = simulateBacteria(1, init, t_eval, 'harvest')
        _, insulin = sim.sol.y
        if i not in timeI:
            timeI[i] = insulin[-1]

    for t, insul in timeI.items():
        cumI = 0
        cumsumI = []
        for i in range(batches + 1):
            cumsumI.append(cumI)
            cumI += insul
        t_eval = np.arange(0, t * (batches + 1), t)

        index_just_less_than_7500 = np.searchsorted(t_eval, 7500) - 1

        if index_just_less_than_7500 >= 0 and t_eval[index_just_less_than_7500] < 7500:
            t_values.append(t)
            insulin_values.append(cumsumI[index_just_less_than_7500])
    max_insulin_index = np.argmax(insulin_values)  # Get the index of the maximum insulin value
    max_t = t_values[max_insulin_index]  # Get the corresponding time
    # Plotting
    plt.scatter(t_values, insulin_values, s = 10)
    plt.axvline(x=max_t, linestyle='--', color='red', alpha=0.5, label='t = {}'.format(max_t))
    plt.xlabel('harvest times (minutes)')
    plt.ylabel('Insulin concentration at 7500 minutes')
    plt.title('Insulin Concentration vs. harvest time t')
    plt.legend()
    plt.show()

def main():
    result1()





if __name__ == '__main__':
    main()
    