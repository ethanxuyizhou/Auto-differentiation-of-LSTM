# some useful functions

#!/usr/bin/env python
from __future__ import division
import numpy as np
from xman import *


# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    # TODO add other operation registers

    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu', a)

    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax', a)

    @staticmethod
    def crossEnt(P, T):
        return XManFunctions.registerDefinedByOperator('crossEnt', P, T)

    @staticmethod
    def mean(a):
        return XManFunctions.registerDefinedByOperator('mean', a)

    @staticmethod
    def tanh(a):
        return XManFunctions.registerDefinedByOperator('tanh', a)


    @staticmethod
    def sigmoid(a):
        return XManFunctions.registerDefinedByOperator('sigmoid', a)

    @staticmethod
    def hadamard(a,b):
        return XManFunctions.registerDefinedByOperator('hadamard', a,b)


# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

def relu(X):
    Y = np.copy(X)
    Y[X<0] = 0
    return Y

def softMax(a):
    D = np.exp(a)
    return D/np.reshape(np.sum(D, axis = 1), [-1, 1])

def crossEnt(P, T):
    return np.reshape(np.sum(-np.multiply(T, np.log(P)), axis = 1), [-1, 1])

def mean(A):
    return A.sum()/A.shape[0]

def tanh(A):
    return np.tanh(A)

def sigmoid(A):
    return 1./(1. + np.exp(-A))

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'square':   np.square,
    'mul':      lambda x1, x2: np.dot(x1, x2),
    'relu':     relu,
    'softMax':  softMax,
    'crossEnt': crossEnt,
    'mean':   mean,
    'tanh':     tanh, 
    'sigmoid':  sigmoid,
    'hadamard': np.multiply
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
#
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation
# crossEnt-softMax below.

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta

def _reluhelp(delta, out, x):
    temp = out>0
    result = 1*temp
    return np.multiply(delta, result)

def crossEnt_softMax(delta, out, O2, T):
    A = np.exp(O2)
    B = np.sum(A, axis = 1)

    return np.multiply(delta, np.add(-T, A/np.reshape(B, [-1, 1])))

def dummy(delta, out, O2,T):
    return T

def tanh_b(delta, out, x):
    j = tanh(x)
    return np.multiply(delta, 1. - np.multiply(j, j))

def sigmoid_b(delta, out, x):
    j = sigmoid(x)
    return np.multiply(delta, np.multiply(j, 1. - j))

def hadamard_bL(delta, out, x1, x2):
    return np.multiply(delta, x2)

def hadamard_bR(delta, out, x1, x2):
    return np.multiply(delta, x1)

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'mul':              [lambda delta, out, x1, x2: np.dot(delta, x2.T), lambda delta, out, x1, x2: np.dot(x1.T, delta)],
    'relu':             [lambda delta, out, x: _reluhelp(delta, out, x)],
    'crossEnt-softMax': [crossEnt_softMax, dummy],
    'mean': [lambda delta, out, x: np.add(np.zeros(x.shape), delta/float(x.shape[0]))],
    'tanh':             [tanh_b],
    'sigmoid':          [sigmoid_b],
    'hadamard':         [hadamard_bL, hadamard_bR]
    }

def test_allclose(test_name, real, expected, rtol=1e-7, score=10.0/7):
    try:
        np.testing.assert_allclose(real, expected, rtol=rtol)
    except Exception, e:
        print "LSTM UNIT TESTS %s FAILED" % test_name
        print e

# Unit tests for the functions. Run by `python functions.py`.
if __name__ == '__main__':
    x = np.array([
        [ 0.56868927,  0.88004577,  0.19957348],
        [ 0.15738243,  0.83758796,  0.08086533]])
    y = np.array([
        [ 0.58373435,  0.30384789,  0.96120856],
        [ 0.78027835,  0.34217174,  0.6432494 ]])
    delta = np.array([
        [ 0.66130212,  0.11137438,  0.64986955],
        [ 0.9184291 ,  0.43875828,  0.56809095]])
    # Eval tanh
    expected_tanh_x = np.array([
        [ 0.51439602,  0.70644225,  0.19696538],
        [ 0.15609577,  0.68452937,  0.08068952]])
    test_allclose('Eval tanh', EVAL_FUNS['tanh'](x), expected_tanh_x)
    # Eval sigmoid
    expected_sigmoid_x = np.array([
        [ 0.63846068,  0.70683171,  0.54972842],
        [ 0.5392646 ,  0.69795697,  0.52020532]])
    test_allclose('Eval sigmoid', EVAL_FUNS['sigmoid'](x), expected_sigmoid_x)
    # Eval hadamard
    expreced_x_hadamard_y = np.array([
        [ 0.33196346,  0.26740005,  0.19183174],
        [ 0.12280211,  0.28659893,  0.05201657]])
    test_allclose('Eval hadamard', EVAL_FUNS['hadamard'](x, y), expreced_x_hadamard_y)
    # BP tanh
    test_allclose('BP tanh', BP_FUNS['tanh'][0](delta, expected_tanh_x, x), np.array([
        [ 0.48631942,  0.05579181,  0.62465763],
       [ 0.89605076,  0.23316473,  0.56439223]]))
    # BP sigmoid
    test_allclose('BP sigmoid', BP_FUNS['sigmoid'][0](delta, expected_sigmoid_x, x), np.array([
        [ 0.15264747,  0.02307907,  0.16086031],
        [ 0.22819133,  0.09249597,  0.14179081]]))
    # BP hadamard
    test_allclose('BP hadamard 0', BP_FUNS['hadamard'][0](delta, expreced_x_hadamard_y, x, y), np.array([
        [ 0.38602476,  0.03384087,  0.62466017],
        [ 0.71663035,  0.15013069,  0.36542416]]))
    test_allclose('BP hadamard 1', BP_FUNS['hadamard'][1](delta, expreced_x_hadamard_y, x, y), np.array([
        [ 0.37607542,  0.09801456,  0.12969673],
        [ 0.14454461,  0.36749866,  0.04593886]]))
    print 'done'