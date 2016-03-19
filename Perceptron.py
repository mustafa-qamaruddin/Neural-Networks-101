from math import exp
import numpy

class Perceptron:
    eta = 0.01
    epochs = 100
    b = 0.001
    w = [-b, 0, 0]
    ## constant for feature index in dataset
    fi = 0
    fj = 1
    ## constant for classes to test only ignore rest
    ci = 0
    cj = 1
    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs
        self.b = 0.001
        ## INITIAL WEGHTS
        self.w = [-self.b, 0, 0]
        
    def signum(self, V):
        return 1 / 1 + exp(V)
    
    def activation(self, PHI):
        if PHI > 0:
            return +1
        else:
            return -1
    
    def train(self, data, target):
        counter = 0
        for T in data:
            ## SKIPS CLASSES TO LIMIT PROBLEM TO ONLY TWO CLASSES ##
            if target != self.ci and target != self.cj:
                counter = counter + 1
                continue
            ## END SKIP ##
            X = [1 ,T[self.fj], T[self.fj]] ## Inputs only two features
            #print 'X = ', X
            #print 'W = ', W
            V = numpy.inner(X, self.w)
            #print 'V = ', V
            Y = self.signum(V)
            #print 'Y = ', Y
            Y = self.activation(Y)
            d = target
            #print 'Y = ', Y
            #print 'd = ', target
            #print 'W(n) = ', self.w
            self.w = self.w + self.eta * (d - Y) * numpy.array(X)
            ##print 'W(n+1) = ', self.w
            counter = counter + 1
        
    def test(self, data, target):
        print 'Testing:'
        correct = 0
        wrong = 0
        counter = 0
        for T in data:
            ## SKIPS CLASSES TO LIMIT PROBLEM TO ONLY TWO CLASSES ##
            if target != self.ci and target != self.cj:
                counter = counter + 1
                continue
            ## END SKIP ##
            X = [1 ,T[self.fj], T[self.fj]] ## Inputs only two features
            #print 'X = ', X
            #print 'W = ', W
            V = numpy.inner(X, self.w)
            #print 'V = ', V
            Y = self.signum(V)
            #print 'Y = ', Y
            Y = self.activation(Y)
            d = target
            print 'Y = ', Y
            print 'd = ', target
            if Y == 1 and target == self.ci:
                correct = correct + 1
            else:
                wrong = wrong + 1
            counter = counter + 1
        print correct
        print wrong