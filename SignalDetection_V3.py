
import scipy.stats as stats

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

