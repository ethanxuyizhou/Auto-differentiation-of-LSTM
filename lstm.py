"""
Long Short Term Memory for character level entity classification
"""
#!/usr/bin/env python
from __future__ import division
import argparse
import numpy as np
from utils import *
from xman import *
from autograd import *
from functions import *
import math
import sys
import copy
import time
np.random.seed(0)


class LSTM(object):

    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.max_len = max_len
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.my_xman = self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):


        x = XMan()

        a1 = math.sqrt(6.0/(self.in_size + self.num_hid))

        x.Wi = f.param(name="Wi", default = a1*np.random.uniform(-1., 1., (self.in_size, self.num_hid))) 
        x.Wf = f.param(name="Wf", default = a1*np.random.uniform(-1., 1., (self.in_size, self.num_hid))) 
        x.Wo = f.param(name="Wo", default = a1*np.random.uniform(-1., 1., (self.in_size, self.num_hid))) 
        x.Wc = f.param(name="Wc", default = a1*np.random.uniform(-1., 1., (self.in_size, self.num_hid))) 

        a3 = math.sqrt(6.0/(self.num_hid + self.out_size))
        x.W = f.param(name='W', default = a3*np.random.uniform(-1., 1., (self.num_hid, self.out_size)))


        a2 = math.sqrt(6.0/(self.num_hid + self.num_hid))
        x.Ui = f.param(name="Ui", default = a2*np.random.uniform(-1., 1., (self.num_hid, self.num_hid)))
        x.Uf = f.param(name="Uf", default = a2*np.random.uniform(-1., 1., (self.num_hid, self.num_hid)))
        x.Uo = f.param(name="Uo", default = a2*np.random.uniform(-1., 1., (self.num_hid, self.num_hid)))
        x.Uc = f.param(name="Uc", default = a2*np.random.uniform(-1., 1., (self.num_hid, self.num_hid)))


        x.bi = f.param(name="bi", default = 0.1*np.random.uniform(-1., 1., (self.num_hid,)))
        x.bf = f.param(name="bf", default = 0.1*np.random.uniform(-1., 1., (self.num_hid,)))
        x.bo = f.param(name="bo", default = 0.1*np.random.uniform(-1., 1., (self.num_hid,)))
        x.bc = f.param(name="bc", default = 0.1*np.random.uniform(-1., 1., (self.num_hid, )))
        x.b = f.param(name='b', default = 0.1*np.random.uniform(-1., 1., (self.out_size, )))


        xH = f.input(name="xH", default = np.zeros((1, self.num_hid)))

        xc = f.input(name="xc", default = np.zeros((1, self.num_hid)))

        x.target = f.input(name ='y', default = np.eye(1, self.out_size))

        for t in range (self.max_len):

            inpu = f.input(name='x'+str(t), default = np.random.rand(1, self.in_size))

            i = f.sigmoid(f.mul(inpu, x.Wi) + f.mul(xH, x.Ui) + x.bi)

            F = f.sigmoid(f.mul(inpu, x.Wf) + f.mul(xH, x.Uf) + x.bf)

            o = f.sigmoid(f.mul(inpu, x.Wo) + f.mul(xH, x.Uo) + x.bo)

            c = f.tanh(f.mul(inpu, x.Wc) + f.mul(xH, x.Uc) + x.bc)

            xc = f.hadamard(F, xc) + f.hadamard(i, c)

            xH = f.hadamard(o, f.tanh(xc))

        


        x.O = f.relu(f.mul(xH, x.W) + x.b)
        x.P = f.softMax(x.O)
        x.E = f.crossEnt(x.P, x.target)
        x.loss = f.mean(x.E)
        
        return x.setup()


def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE
    # print(max_len)
    # print(mb_train.num_chars)
    # print(num_hid)
    # print(mb_train.num_labels)


    print "done"

    # train
    print "training..."
    # get default data and params

    ls = Autograd(lstm)
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    
    best_dict = value_dict
    best_loss = float(sys.maxint)

    opseq = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    run_time = 0

    for i in range(epochs):
        s_time = time.time()

        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            
            #save the train loss

            size = e.shape[0]

            for a in range ((e.shape[1])):
                value_dict['x' + str(e.shape[1] - 1 - a)] = e[:, a, :]


            value_dict['y'] = l

            value_dict['xH'] = np.zeros((size, num_hid))
            value_dict['xc'] = np.zeros((size, num_hid))

            ls = Autograd(lstm.my_xman)
            value_dict = ls.eval(opseq, value_dict)
            grads = ls.bprop(opseq, value_dict, loss = np.float_(1.))

            for rname in grads:
                if lstm.my_xman.isParam(rname):
                    value_dict[rname] -= lr * grads[rname]

            train_loss = np.append(train_loss, value_dict['loss'])
        
        end_time = time.time()
        run_time += (end_time - s_time)                           
        # validate

        total_valid_loss = 0

        for (idxs,e,l) in mb_valid:

            size = e.shape[0]

            for a in range ((e.shape[1])):
                value_dict['x' + str(e.shape[1] - 1 - a)] = e[:, a, :]

            value_dict['y'] = l


            value_dict['xH'] = np.zeros((size, num_hid))
            value_dict['xc'] = np.zeros((size, num_hid))

            ls = Autograd(lstm.my_xman)
            value_dict = ls.eval(opseq, value_dict)
            valid_loss = value_dict['loss']
            #print(total_valid_loss)
            #TODO prepare the input and do a fwd pass over it to compute the loss

        current_valid = np.mean(valid_loss)

        if (current_valid < best_loss):
            best_dict = copy.deepcopy(value_dict)
            best_loss = current_valid

        #TODO compare current validation loss to minimum validation loss
        # and store params if needed
    print "done"
    #print(run_time/epochs)

    #write out the train loss
    np.save(train_loss_file, train_loss)    
    
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        size = e.shape[0]

        for a in range ((e.shape[1])):
            best_dict['x' + str(e.shape[1] - 1- a)] = e[:, a, :]

        best_dict['y'] = l

        best_dict['xH'] = np.zeros((size, num_hid))
        best_dict['xc'] = np.zeros((size, num_hid))

        ls = Autograd(lstm.my_xman)
        best_dict = ls.eval(opseq, best_dict)
        ouput_probabilities =  best_dict['P']
        print(best_dict['loss'])

    np.save(output_file, ouput_probabilities)
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    #np.save(output_file, ouput_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)