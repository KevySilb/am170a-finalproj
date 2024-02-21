import pandas as pd
from matplotlib import pyplot as plt
import scipy.integrate as itg
'''1mg/hour per mL for a bacterial solution containing 0.1g of dry bacteria per mL'''

class simulateBacteria:
    """
    alpha : bacteria growth rate
    beta  : insulin potency
    kappa : carrying capacity 

    self.sol.y contants (bacteria(t), insulin(t))
    """
    def __init__(self, kappa, init, t_points) -> None:
        self.a, self.b, self.g = 0.0355, 1.68, 5.41e-08
        self.k = kappa
        self.x0, self.y0 = init
        self.t_points = t_points
        self.sol = self.simulate()
    
    def func(self, t, r):
        x, y = r
        dx_dt = self.a * x * (1-(x/self.k)) - self.b * x * y 
        dy_dt = self.g * x
        return dx_dt, dy_dt
    
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
    



if __name__ == '__main__':
    main()
    