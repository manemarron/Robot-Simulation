# -*- coding: utf8 -*-
import numpy as np
from utils import rk4


class Robot:
    def __init__(self, km, ke, mp, mw, ip, iw, l, r, res, g):
        """
        Physical constants that describe the robot's system are initialized:
        Total mass, height, wheel radius, etc.

        Parameters
        ----------
        :param km : float
            Motors' torque constant
        :param ke : float
            Back EMF constant (counter-electromotive force)
        :param mp : float
            Robot's mass
        :param  mw : float
            Wheels' mass
        :param ip : float
            Robot's momentum
        :param iw : float
            Wheels' momentum
        :param l  : float
            Distance between wheels' gravity center and robot's gravity center
        :param r  : float
            Wheels' radius
        :param res  : float
            Nominal Terminal resistance
        :param g  : float
            Gravity acceleration constant
        """
        self.km = km
        self.ke = ke
        self.Mp = mp
        self.Mw = mw
        self.Ip = ip
        self.Iw = iw
        self.l = l
        self.r = r
        self.R = res
        self.g = g

        beta = 2 * mw + 2 * iw / r ** 2 + mp
        alpha = ip * beta + 2 * mp * l ** 2 * (mw + iw / r ** 2)

        self._init_a(alpha, beta)
        self._init_b(alpha, beta)

    def _init_a(self, alpha, beta):
        """
        Initializes the state transition matrix of the robot's system
        :param alpha: float
        :param beta: float
        """
        a22 = (2 * self.km * self.ke *
               (self.Mp * self.l * self.r - self.Ip - self.Mp * self.l ** 2) / (self.R * alpha * self.r ** 2))
        a23 = self.Mp ** 2 * self.g * self.l ** 2 / alpha
        a42 = 2 * self.km * self.ke * (self.r * beta - self.Mp * self.l) / (self.R * alpha * self.r ** 2)
        a43 = self.Mp * self.g * self.l * beta / alpha

        self.A = np.matrix([
            [0, 1, 0, 0],
            [0, a22, a23, 0],
            [0, 0, 0, 1],
            [0, a42, a43, 0],
        ])

    def _init_b(self, alpha, beta):
        """
        Initializes the control matrix of the robot's system
        :param alpha: float
        :param beta: float
        """
        b2 = 2 * self.km * (self.Ip + self.Mp * self.l ** 2 - self.Mp * self.l * self.r) / (self.R * self.r * alpha)
        b4 = 2 * self.km * (self.Mp * self.l - self.r * beta) / (self.R * self.r * alpha)

        self.B = np.matrix([0, b2, 0, b4]).T

    def dynamics(self, state):
        return self.A * state

    def integrate(self, n, state, dt):
        history = np.array([state])
        for i in range(n):
            state = rk4(self.dynamics, state, i, dt)
            history = np.append(history, [state], axis=0)
        return history

    @staticmethod
    def _get_gyroscope(theta):
        """
        Gets angle with noise component
        :param theta: float
        :rtype : float
        """
        return theta + np.random.randn() * 0.01

    @staticmethod
    def _get_accelerometer(a):
        """
        Gets acceleration with noise.
        :param a: float
        :rtype: float
        """
        return a + np.random.randn() * 0.01
