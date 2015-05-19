# -*- coding: utf8 -*-

class PID:
    def __init__(self, kp, kd, ki, angle, dt):
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki
        self.angle = angle
        self.dt = dt

        self.prev_error = 0
        self.Ci = 0

    def get_error(self, new_angle):
        return new_angle - self.angle

    def execute_pid(self, new_angle):
        error = self.get_error(new_angle)
        de = error - self.prev_error

        cp = error
        self.Ci += error * self.dt
        cd = 0
        if self.dt > 0:
            cd = de/self.dt

        self.prev_error = error
        return (self.Kp * cp) + (self.Ki * self.Ci) + (self.Kd*cd)
