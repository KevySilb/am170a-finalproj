import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as itg
from matplotlib import gridspec
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
    init = (100, 0)
    params_harvest = (1e4, 0)
    params_extract = (1e4, 2e-1)
    t_max = 250
    t_eval = np.arange(0, t_max, 10)

    sim_harvest = simulateBacteria(params_harvest, init, t_eval)
    bac_harv, insul_harv = sim_harvest.sol.y

    # feed insulin value at t= 250 into the initial conditions recursively 
    sim_harvest1 = simulateBacteria()
    plt.plot(t_eval, insul_harv)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()