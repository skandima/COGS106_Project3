#Import needed library
import scipy.stats as stats

#Implement class based on signal detection theory
class SignalDetection:
    #Class constructor
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        #signal detection theory variables
        self.hits = hits #-0.0000001 needed for problems encountered with using stats.norm.ppf
        self.misses = misses
        self.falseAlarms = falseAlarms  #-0.0000001 needed for problems encountered with using stats.norm.ppf
        self.correctRejections = correctRejections
        #Note: hits and false alarms only effected as they are the numerator and denominator in the rate equations

    #class method to get the hit rate: hits/total signal trials
    def hit_rate(self):
        return self.hits / (self.hits + self.misses)

    #class method to get the false alarm rate: false alarms/total noise trials
    def falseAlarm_rate(self):
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)

    #class method to calculate d': Z(H) - Z(FA), Z = stats.norm.ppf
    def d_prime(self):
        #get hit rate
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        #get z values
        z_hit = stats.norm.ppf(hit_rate) 
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        #return d'
        return z_hit-z_falseAlarm

    #class method to get the criterion: -0.5*(Z(H) + Z(FA)), Z = stats.norm.ppf
    def criterion(self):
        #get hit rates
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        #get Z values
        z_hit = stats.norm.ppf(hit_rate)
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        #return criterion
        return -0.5*(z_hit + z_falseAlarm)

