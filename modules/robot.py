# -*- coding: utf8 -*-
import numpy as np


class Robot:
    def __init__(self, km, ke, Mp, Mw, Ip, Iw, l, r, R, g):
        """
        Se inicializan las constantes físicas del chasis,
        llantas y motores del sistema.

        Parámetros:
        - km: constante de torque de los motores
        - ke: constante de fuerza contraelectromotriz
        - Mp: masa del chasis (incluyendo motores y llantas)
        - Mw: masa de la llanta
        - Ip: momento de inercia del robot
        - Iw: momento de inercia de las llantas
        - l:  distancia entre el centro de gravedad de la llanta y
              el centro de gravedad del robot
        - r:  radio de la llanta
        - R:  Resistencia nominal
        - g:  constante de aceleración de la gravedad
        """
        self.km = km
        self.ke = ke
        self.Mp = Mp
        self.Mw = Mw
        self.Ip = Ip
        self.Iw = Iw
        self.l = l
        self.r = r
        self.R = R
        self.g = g

        beta = 2*Mw + 2*Iw/r**2 + Mp
        alpha = Ip*beta + 2*Mp*l**2*(Mw + Iw/r**2)

        self._init_A(km, ke, Mp, Mw, Ip, Iw, l, r, R, g, alpha, beta)
        self._init_B(km, ke, Mp, Mw, Ip, Iw, l, r, R, g, alpha, beta)

    def _init_A(self, km, ke, Mp, Mw, Ip, Iw, l, r, R, g, alpha, beta):
        """
        Se genera la matriz de dinámica del sistema
        """
        a22 = 2*km*ke*(Mp*l*r - Ip - Mp*l**2)/(R*alpha*r**2)
        a23 = Mp**2 * g * l**2 / alpha
        a42 = 2*km*ke*(r*beta - Mp*l)/(R*alpha*r**2)
        a43 = Mp*g*l*beta/alpha

        self.A = np.matrix([
            [0,   1,   0, 0],
            [0, a22, a23, 0],
            [0,   0,   0, 1],
            [0, a42, a43, 0],
        ])

    def _init_B(self, km, ke, Mp, Mw, Ip, Iw, l, r, R, g, alpha, beta):
        """
        Se genera la matriz de control del sistema
        """
        b2 = 2*km*(Ip + Mp*l**2 - Mp*l*r)/(R*r*alpha)
        b4 = 2*km*(Mp*l - r*beta)/(R*r*alpha)

        self.B = np.matrix([0, b2, 0, b4]).T

    def dinamica(self, state, t, u=0):
        """
        Define la dinámica del sistema a partir de un estado
        """
        return self.A*state + self.B*u





class RobotConFiltro(Robot):
    """
    Dentro de esta clase se abstrae el comportamiento de un
    robot bípedo utilizando un filtro de Kalman
    """
    def __init__(self, km, ke, Mp, Mw, Ip, Iw, l, r, R, g):
        super.__init__(km, ke, Mp, Mw, Ip, Iw, l, r, R, g)

    def _init_filtro_kalman(self):
        self.P = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.Q = np.matrix(
            [0, 0, 0, 0],
            [0, 0.01, 0, 0],
            [0, 0, 0.01, 0],
            [0, 0, 0, 0]
        )
        self.Var_medicion = 0.002

    def filtroKalman(self, state, t, u=0):
        """
        Implementación del filtro de Kalman
        """
        # Etapa de predicción
        X = self.dinamica(state, t, u)
        self.P = (self.A * self.P * self.A.T) + self.Q
