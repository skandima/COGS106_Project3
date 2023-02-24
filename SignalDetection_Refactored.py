
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm



class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.__hits = hits 
        self.__misses = misses
        self.__falseAlarms = falseAlarms 
        self.__correctRejections = correctRejections

    def hit_rate(self):
        return self.__hits / (self.__hits + self.__misses)

    def falseAlarm_rate(self):
        return self.__falseAlarms / (self.__falseAlarms + self.__correctRejections)

    def d_prime(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate) 
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return z_hit-z_falseAlarm

    def criterion(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate)
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return -0.5*(z_hit + z_falseAlarm)

    def __add__(self, other):
        return SignalDetection(self.__hits + other.__hits, self.__misses + other.__misses, 
                               self.__falseAlarms + other.__falseAlarms, 
                               self.__correctRejections + other.__correctRejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.__hits * scalar, self.__misses * scalar, 
                               self.__falseAlarms * scalar, 
                               self.__correctRejections * scalar)

    def plot_roc(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        plt.plot([0, falseAlarm_rate, 1], [0, hit_rate, 1], 'b')
        plt.scatter(falseAlarm_rate, hit_rate, c='r')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Alarm rate')
        plt.ylabel('Hit rate')
        plt.title('Receiver Operating Characteristic (ROC) curve')
        plt.show()

sd = SignalDetection(10, 30, 10, 5)
sd.plot_roc()


