# -*- coding: utf8 -*-
import numpy as np


def rk4(f, y, t, dt):
    """
    Integrates an ordinary differential equation using the order 4 Runge-Kutta method
    :param f: function
        Function to be integrated
    :param y: array
        Array containing previous state
    :param t: float
        Number representing current value of time
    :param dt: float
        Represents the time step for the integration
    :return: array
        Array containing the next state
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * k1 * dt, t + 0.5 * dt)
    k3 = f(y + 0.5 * k2 * dt, t + 0.5 * dt)
    k4 = f(y + k3 * dt, t + dt)

    res = y + float(1) / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return res

