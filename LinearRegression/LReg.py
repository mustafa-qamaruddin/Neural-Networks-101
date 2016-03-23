from math import exp
import numpy

class LinearRegression:
    err  = 0
    epoch = 1
    lamda = 0.1
    b = 0.0001
    w = [0, 0]
    I = numpy.eye(2)
    print I
    ## constant for feature index in dataset
    fi = 0
    fj = 1
    ## constant for classes to test only ignore rest
    ci = 0
    cj = 1
    def __init__(self, err, epochs,lamda):
        self.eta = err
        self.epoch = epochs
        self.lamda = lamda
        ## INITIAL WEGHTS
        self.w = [0, 0]
        self.I = numpy.eye(2)
    def signum(self, PHI):

        if PHI > 0:
            return 0
        else:
            return 1
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
        ############################Training LinearRegression##############################
    def train(self, data):
            M = self.Get_Mean(data)
            MX = self.Get_Max(data)
            ee = []
            counter = 0
            for T in data:

                X = T[0] ## Input
                d = T[1] ## Desired Response
                ## SKIPS CLASSES TO LIMIT PROBLEM TO ONLY TWO CLASSES ##
                if d != self.ci and d != self.cj:
                    counter = counter + 1
                    continue
                ## END SKIP ##
                X = [(X[self.fi] - M[0])/MX[0], (X[self.fj]-M[1])/MX[1]] ## Inputs only two features
                RXX = -1 * numpy.inner(X,X)
                #print RXX
                LI = self.lamda * self.I
                INV =  numpy.linalg.inv(RXX + LI)
                rdx = -1 * numpy.inner(X,d)
                self.w =  numpy.inner(INV,rdx)
                #print self.w
            ############################Testing LinearRegression##############################
    def test(self, data):
        print 'Testing:'
        M = self.Get_Mean(data)
        MX = self.Get_Max(data)
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
            X = [(X[self.fi] - M[0])/MX[0], (X[self.fj]-M[1])/MX[1]] ## Inputs only two features
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