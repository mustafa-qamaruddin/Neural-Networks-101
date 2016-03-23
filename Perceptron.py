from math import exp
import numpy

class Perceptron:
    eta = 0.0001
    epochs = 100
    thres = 1e-2
    b = 0.0001
    w = [-b, 0, 0]
    ## constant for feature index in dataset
    fi = 0
    fj = 1
    ## constant for classes to test only ignore rest
    ci = 0
    cj = 1
    def __init__(self, eta, epochs,Thresh):
        self.eta = eta
        self.epochs = epochs
        self.thres = Thresh
        self.b = 0.001
        ## INITIAL WEGHTS
        self.w = [self.b, 0, 0]
        
    def sigmoid(self, V):
        return 1 / 1 + exp(V)

    def signum(self, PHI):
            if PHI > 0:
                return 0
            else:
                return 1 
    
    def activation(self, PHI):
        if PHI > 0:
            return +1
        else:
            return -1

    def Get_Mean(self,data):
            sum1 = 0
            sum2 = 0
            for T in data:
                X = T[0] ## Input
                d = T[1] ## Desired Response
                sum1 += X[self.fi]
                sum2 += X[self.fj]
            mean0 = [sum1/150,sum2/150,0]
            return mean0

    def Get_Max(self,data):
        max1 = 0
        max2 = 0
        counter = 0
        for T in data:
            X = T[0] ## Input
            d = T[1] ## Desired Response
            if counter < 50:
               max1 = max(max1,X[self.fi])
               max2 = max(max2,X[self.fj])
            counter = counter + 1
            Maxi = [max1 , max2]
            return Maxi

    def train(self, data):
        M = self.Get_Mean(data)
        MX = self.Get_Max(data)
        ee = []
        for epo_i in range(0, self.epochs):
            counter = 0
            for T in data:

                X = T[0] ## Input
                d = T[1] ## Desired Response
                ## SKIPS CLASSES TO LIMIT PROBLEM TO ONLY TWO CLASSES ##
                if d != self.ci and d != self.cj:
                    counter = counter + 1
                    continue
                ## END SKIP ##
                X = [1 ,(X[self.fi] - M[0])/MX[0], (X[self.fj]-M[1])/MX[1]] ## Inputs only two features
                #print X
                V = numpy.inner(self.w, X)
                Y = V
               # Y = self.activation(Y)
                e = d - Y
                self.w = self.w + self.eta * e * numpy.array(X)
                ee.append(e)
                #print 'W(n+1) = ', self.w
                counter = counter + 1
            mse =  numpy.mean(0.5 * (ee[epo_i]*ee[epo_i]))
            if mse < self.thres:
                break


    def test(self, data):
        print 'Testing:'
        correct = 0
        wrong = 0
        counter = 0
        for T in data:
            X = T[0] ## Input
            d = T[1] ## Desired Response            
            ## SKIPS CLASSES TO LIMIT PROBLEM TO ONLY TWO CLASSES ##
            if d != self.ci and d != self.cj:
                counter = counter + 1
                continue
            ## END SKIP ##
            X = [1 ,X[self.fj], X[self.fj]] ## Inputs only two features
            #print 'X = ', X

            V = numpy.inner(self.w, X)
            #print 'V = ', V
            Y = self.signum(V)
            #Y = self.activation(Y)
            print 'Y = ', Y
            print 'd = ', d
            if Y == d:
                correct = correct + 1
            else:
                wrong = wrong + 1
            counter = counter + 1
        print 'correct=', correct
        print 'wrong=', wrong