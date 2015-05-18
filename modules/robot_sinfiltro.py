# -*- coding: utf8 -*-
__author__ = 'manemarron'
import numpy as np

from robot import Robot

class RobotSinFiltro(Robot):
    """
    This class abstracts the behaviour of a biped self-balancing robot
    that does not use a Kalman Filter.
    """
    def __init__(self, km, ke, Mp, Mw, Ip, Iw, l, r, R, g):
        super.__init__(km, ke, Mp, Mw, Ip, Iw, l, r, R, g)

    def RK4(self, f, y, t, dt):
        """
        Implementación del método de integración de
        Runge-Kutta de orden 4
        """
        k1 = f(y, t)
        k2 = f(y + 0.5*k1*dt, t+0.5*dt)
        k3 = f(y + 0.5*k2*dt, t+0.5*dt)
        k4 = f(y + k3*dt, t+dt)

        return y + float(1)/6*dt*(k1+2*k2+2*k3+k4)

    def integrar(self, n, state, dt):
        """
        Se evoluciona el sistema un número n de veces a partir
        de un estado inicial dado. No toma en cuenta sensores
        ni control.
        """
        historico = np.array([state])
        for i in range(n):
            state = self.RK4(self.dinamica, state, i, dt)
            historico = np.append(historico, [state], axis=0)
        return historico

    def integrarConSensores(self, n, state, dt):
        historico = np.array([state])
        sensor = np.array(
            [np.array([
                [0, 0],
                [0, self.sensorGiroscopio(state[2])],
                [0, 0]]
            )]
        )
        for i in range(n):
            y = self.dinamica(state, i, dt)
            state = self.RK4(self.dinamica, state, i, dt)
            historico = np.append(historico, [state], axis=0)

            theta = self.sensorGiroscopio(state[3])
            ace = self.sensorAcelerometro(y[1])
            sensor = np.append(
                sensor,
                [np.array(
                    [0, 0],
                    [ace, theta],
                    [0, 0]
                )],
                axis=0
            )
        return historico, sensor

    def sensorGiroscopio(self, theta):
        thetaM = theta + np.random.randn()*0.01
        return thetaM

    def sensorAcelerometro(self, a):
        aM = a + np.random.randn()*0.01
        return aM