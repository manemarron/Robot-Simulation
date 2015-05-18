# -*- coding: utf8 -*-
from robot import *
from pid import PID


class RobotWithController(Robot):
    """
    Subclass of Robot to abstract PID behaviour
    """

    def __init__(self, km, ke, mp, mw, ip, iw, l, r, res, g, pid_kp, pid_ki, pid_kd, pid_angle):
        """
        See Robot class for complete documentation on robot's system constants
        :param pid_kp: float
            Proportional constant of PID controller
        :param pid_ki: float
            Integral constant of PID controller
        :param pid_kd: float
            Derivative constant of PID controller
        :param pid_angle: float
            Angle for the robot to stay balanced
        """
        Robot.__init__(self, km, ke, mp, mw, ip, iw, l, r, res, g)
        self.kp = pid_kp
        self.ki = pid_ki
        self.kd = pid_kd
        self.pidAngle = pid_angle

        self.u = 0

    def dynamics(self, state):
        return self.A * state + self.B * self.u

    def integrate(self, n, state, dt):
        pid = PID(kp=self.kp, ki=self.ki, kd=self.kd, angle=self.pidAngle, dt=dt)
        history = np.array([state])
        theta = Robot._get_gyroscope(state[2])
        for i in range(n):
            self.u = pid.execute_pid(theta)
            state = rk4(self.dynamics, state, i, dt)
            history = np.append(history, [state], axis=0)

            theta = Robot._get_gyroscope(state[2])
        return history

    def execute_kalman_filter(self, n, state, dt):
        pid = PID(kp=self.kp, ki=self.ki, kd=self.kd, angle=self.pidAngle, dt=dt)
        history = np.array([state])
        theta = self._get_gyroscope(state[2])
        x = state
        h = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        p = np.matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        q = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        r = np.matrix([
            [0.01, 0],
            [0, 0.01]
        ])
        for i in range(n):
            z = (h * x) + np.matrix([[np.random.randn()*0.01], [np.random.randn()*0.01]])
            u = pid.execute_pid(theta)
            x = (self.A * x) + (self.B * u)
            p = (self.A * p * self.A.T) + q

            y = z - (h * x)
            s = (h * p * h.T) + r
            k = p * h.T * np.linalg.inv(s)
            x = x + (k * y)
            p = (np.eye(4) - (k * h))*p
            history = np.append(history, [x], axis=0)

        return history
