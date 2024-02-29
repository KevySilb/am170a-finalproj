import numpy as np 
from matplotlib import pyplot as plt
import pygame as pg
import sys
import scipy.integrate as itg

# -*- coding: utf-8 -*-
"""
@author: Kevin
"""
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
    
class GUI:

    def __init__(self):
        pg.display.set_caption('NAME GOES HERE')
        self.screen = pg.display.set_mode(RES)
        self.clock = pg.time.Clock()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                self.click = True
                self.x, self.y = event.pos
                self.seedSubmatrix(self.x, self.y)
            if event.type == pg.MOUSEBUTTONUP:
                self.click = False
            if event.type == pg.MOUSEMOTION:
                if self.click:
                    self.x, self.y = event.pos
                    self.seedSubmatrix(self.x, self.y)

    def run(self):
        while True:
            pg.display.update()
            self.clock.tick(60)
            self.check_events()
            
if __name__ == "__main__":
    N = 600
    pg.init()
    RES = WIDTH, HEIGHT = (N, N)
    app = GUI()
    app.run()