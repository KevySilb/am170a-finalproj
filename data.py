import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as itg
'''1mg/hour per mL for a bacterial solution containing 0.1g of dry bacteria per mL'''

class simulateBacteria:
    """
    bacteria growth rate = 0.0355
    insulin potency = 1.68
    insulin production rate 5.41e-08 mg insulin / (per million bacteria * min)
    self.k = carrying capacity F1 (100), F2 (1,000), F3 (10,000)
    self.a = rate of insulin extraction

    self.sol.y contants (bacteria(t), insulin(t))
    """
    def __init__(self, params, init, t_points) -> None:
        self.k, self.a = params
        self.x0, self.y0 = init
        self.t_points = t_points
        # cumulative sum of insulin extracted 
        self.Insulin = 0
        # a history of extracted insulin per time step 
        self.cumsumInsulin = []
        # the solution of the ivp
        self.sol = self.simulate()
    
    def func(self, t, r):
        """
        x1 represents millions of bacteria in solution at a particular point in time 
        x2 represents mg of insulin per mL
        """
        x1, x2 = r
        # insulin extraction is proportional to the amount of insulin in solution (no insulin = no extraction)
        extractedInsulin = self.a * x2
        # running total of extracted insulin
        self.Insulin += extractedInsulin
        # append new total to list for plotting 
        self.cumsumInsulin.append(self.Insulin)
        
        # the ivp:
        dx1_dt = 0.0355 * x1 * (1-(x1/self.k)) - 1.68 * x1 * x2 
        dx2_dt = 5.41e-08 * x1 - extractedInsulin
        return dx1_dt, dx2_dt
    
    def simulate(self):
        # time interval
        t0 = self.t_points[0]
        tf = self.t_points[-1]
        
        # initial conditions
        init = (self.x0, self.y0)
        
        result = itg.solve_ivp(self.func, [t0, tf], init, t_eval=self.t_points)
        return result

def main():
    # get data
    data = pd.read_csv('data_with_insulin.txt')
    # initial conditions (100 million bacteria, 0 insulin)
    init = (100, 0)
    # define parameters (10000x sugar concentration, rate of insulin extraction))
    params = (10000, .3)
    t_max = 1250
    t_eval = np.arange(0, t_max, 10)

    sim1 = simulateBacteria(params, init, t_eval)
    sim2 = simulateBacteria((10000, 0), init, t_eval)
    bac, insul = sim1.sol.y
    bac2, insul2 = sim2.sol.y
    # plt.plot(t_eval, bac, label = r'Bacteria')
    # plt.plot(t_eval, insul, label = 'mg insulin')
    # plt.plot(t_eval, bac2, label = r'$N_0$')
    plt.plot(t_eval, insul2, label = r'$I_0$')
    plt.plot(np.linspace(0, t_max, len(sim1.cumsumInsulin)), sim1.cumsumInsulin, label = r'Insulin Extracted')
    plt.xlabel('minutes')
    plt.ylabel('log(population density)')
    plt.yscale('log')
    plt.title('Insulin extraction scheme')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
    