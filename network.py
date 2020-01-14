import tushare as ts
import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import copy
'
def init_para(dim):
    network = {}
    for i in range(1,len(dim)):
        network['L%dW'%(i)] = np.random.normal(0,(dim[i-1]+dim[i]),(dim[i-1], dim[i]))
        #network['L%dW'%(i)] = np.zeros((dim[i-1], dim[i]))+1
        #network['L%db'%(i)] = np.random.normal(0,(dim[i-1]+dim[i])/4096,(1, dim[i]))
        network['L%db'%(i)] = np.zeros((1, dim[i]))
    return network
def test(x,y, network, act_fun):
    dim, layers = sizeof(network)
    for i in range(1, layers+1):
        x = np.dot(x, network['L%dW'%(i)]) + network['L%db'%(i)]
        if act_fun[i-1]:
            softplus(x)
        x[np.isnan(x)] = 0
    y = np.array(y, ndmin = 2).T - x
    return x, (y*y).mean(axis=0)
def softplus(A):
    index1 = A <= -36.75
    index2 = (A < 33.28) * np.logical_not(index1)
    A[index1] = 0
    A[index2] = np.log(np.exp(A[index2]) + 1)
def Der_softplus(A, B, c = True):
    B[B >= 36.75] = 1
    B[B <= -33.28] = np.inf
    index = (B > -33.28) * (B < 36.75)
    B[index] = 1 - np.exp(-B[index])
    B[np.abs(B) < 0.00001] = np.inf    
    A = A / B


def train(x_data, y_data, dim = [1,1], network = None, act_fun=None, tra_times = 100, hit = 0.00001):
    if network == None:
        if dim == None:
            dim = [1,1]
        network = init_para(dim)
        layers = len(dim) - 1
    else:
        dim, layers = sizeof(network)
    if act_fun == None:
        act_fun = [None for i in range(1,len(dim))]
    
    fit = 0
    fail = 0
    fitter = 1
    network_old = 0
    cost_old = float("inf")
    jump = True
    
    for s in range(0, tra_times):
        if jump:
            cost = x_data
            for i in range(1,layers+1):
                cost = np.dot(cost, network['L%dW'%(i)]) + network['L%db'%(i)]
                if act_fun[i-1]:
                    softplus(cost)
                cost[np.isnan(cost)] = 0
            cost = cost - y_data
            cost = np.mean((cost*cost).mean(axis=1))
            if cost > cost_old:
                network = copy.deepcopy(network_old)
                fail +=1
                #if fail % 10 == 0:
                #    fitter *= 2
                if fail > 100:
                    break
            else:
                network_old = copy.deepcopy(network)
                cost_old = cost
                #fitter += 2
        
        group_index = random.sample(range(0,len(y_data)), 14)
        values = [np.array(x_data[group_index], ndmin = 2)]

        for i in range(1,layers+1):
            values.append(np.dot(values[i-1], network['L%dW'%(i)])+network['L%db'%(i)])
            if act_fun[i-1]:
                softplus(values[-1])
        target = np.array(y_data[group_index], ndmin = 2).T
        cost = target - values[-1]
        if  np.mean((cost*cost).mean(axis=1)) <= hit:
            fit +=1
            jump = False
            #fitter *= 0.5
            if fit > 100:
                break
            continue
        else:
            jump = True

        for i in range(layers,0,-1):
            step = cost
            if act_fun[i-1]:
                Der_softplus(step, values[i])
            step = np.array([np.sign(( step*cost*cost ).sum(axis=0))], ndmin = 2)
            step[np.isnan(step)] = 0
            step[np.abs(step) > 200000] = 0
            network['L%db'%(i)] = network['L%db'%(i)] + step# * fitter
            values[i] = np.dot(values[i-1], network['L%dW'%(i)]) + network['L%db'%(i)]
            if act_fun[i-1]:
                softplus(values[i])
            cost = target - values[i]
            print(np.shape(target))
            print(np.shape(values[i]))
            for j in range(0,len(network['L%dW'%(i)])):
                step = cost / np.array([ values[i-1][:,j] ], ndmin = 2).T
                step[np.isnan(step)] = 0
                step[np.abs(step) > 200000] = 0
                if act_fun[i-1]:
                    Der_softplus(step, values[i])
                step = np.array([np.sign(( step*cost*cost ).sum(axis=0))], ndmin = 2)
                step[np.isnan(step)] = 0
                step[np.abs(step) > 200000] = 0
                network['L%dW'%(i)][j,:] = network['L%dW'%(i)][j,:] + step * fitter
                values[i] = np.dot(values[i-1], network['L%dW'%(i)]) + network['L%db'%(i)]
                if act_fun[i-1]:
                    softplus(values[i])
                cost = target - values[i]
            '''step = np.linalg.pinv(values[i-1])
            step = np.dot(step, target)
            step[np.isnan(step)] = 0
            step[np.abs(step) > 200000] = 0
            network['L%dW'%(i)] = (network['L%dW'%(i)] * fitter + step) / (fitter + 1)
            values[i] = np.dot(values[i-1], network['L%dW'%(i)]) + network['L%db'%(i)]
            if act_fun[i-1]:
                softplus(values[i])
            cost = target - values[i]'''

            if i == 1:
                break
            
            new_target = values[i-1]
            for j in range(0,len(network['L%dW'%(i)])):
                step = cost / np.array([ network['L%dW'%(i)][j,:] ], ndmin = 2)
                step[np.isnan(step)] = 0
                step[np.abs(step) > 200000] = 0
                if act_fun[i-1]:
                    Der_softplus(step, values[i], False)
                step = np.array([np.sign(( step*cost*cost ).sum(axis=1))], ndmin = 2).T
                step[np.isnan(step)] = 0
                step[np.abs(step) > 200000] = 0
                new_target[:,j] = new_target[:,j] + step.flatten() * fitter
                values[i] = np.dot(values[i-1], network['L%dW'%(i)]) + network['L%db'%(i)]
                if act_fun[i-1]:
                    softplus(values[i])
                cost = target - values[i]
            target = new_target
            cost = target - values[i-1]
            
            '''step = np.linalg.pinv(network['L%dW'%(i)])            
            step = np.dot(target, step)
            step[np.isnan(step)] = 0
            step[np.abs(step) > 200000] = 0
            target = (values[i-1] * fitter + step) / (fitter + 1)
            cost = target - values[i-1]'''

    cost = x_data
    for i in range(1,layers+1):
        cost = np.dot(cost, network['L%dW'%(i)]) + network['L%db'%(i)]
        if act_fun[i-1]:
            softplus(cost)
        cost[np.isnan(cost)] = 0
    plt.plot(cost)
    plt.plot(y_data, color = 'r')
    plt.show()
    cost = cost - y_data
    cost = np.mean((cost*cost).mean(axis=1))
    if cost > cost_old:
        network = copy.deepcopy(network_old)
    else:
        cost_old = cost

    print(cost_old)
    return network


net = train(x_data, y_data, dim = dim, network = None, act_fun=act_fun, tra_times = 0, hit = 0.1)