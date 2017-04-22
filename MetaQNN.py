# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:06:08 2017

@author: jjgutier
"""

import statespace_test as SA

import tensorflow as tf
import numpy as np
import random



def Q_learning(NUMSTATES,NUMACTIONS,M,epsilon,K,statespace,A):
    replay_memory = []
    Q = tf.Variable(tf.constant(0.5, shape=[NUMSTATES, NUMSTATES]),name='Q')
    for episode in range(1,M):
        S,U = SAMPLE_NEW_NETWORK(epsilon,Q,statespace,A)
        accuracy = TRAIN(S,statespace)
        replay_memory.append((S,U,accuracy))
        for memory in range(K):
            S_sample, U_sample, accuracy_sample = replay_memory[int(random.uniform(0,len(replay_memory)))]
            Q = UPDATE_Q_VALUES(Q,S_sample,U_sample,accuracy_sample)
        #print(replay_memory)
    return Q,S,U

def SAMPLE_NEW_NETWORK(epsilon, Q,statespace,A):
    S = [statespace[0][1]]
    U = [0]
    while U[-1] != 562:
        alpha = random.uniform(0,1)
        if alpha > epsilon:

            u = tf.arg_max(Q[0],dimension=1)
            sprime = TRANSITION(S[-1],u)
        else:
            u = A[S[-1]][int(random.uniform(0,len(A[S[-1]])-1))]
            sprime = TRANSITION(S[-1],u)
        U.append(u)
        #print(S)
        if u != 562:
            S.append(sprime)
            
    return S,U

def UPDATE_Q_VALUES(Q,S,U,accuracy):
    global alpha
    print(S,U)
    Q[S[-1],U[-1]].assign((1-alpha)*Q[S[-1],U[-1]] + alpha*accuracy)
    for i in reversed(range(len(S)-2)):
        Q[S[i],U[i]].assign((1-alpha)*Q[S[i],U[i]] + alpha*Q[S[i+1],U[i]])
    return Q


def TRAIN(S,statespace):
    inputimage = 0.5
    
    MetaQNN(images, keep_prob, S,statespace)
        
    return 0.5


def TRANSITION(s,u):
    sprime = u
    return sprime


def MetaQNN(images, keep_prob):
    with tf.variable_scope('conv1') as scope:
            kernel = __var_on_cpu_mem('weights',
                                  [FILTER_SIZE,
                                   FILTER_SIZE,
                                   FILTER_SIZE,
                                   IN_CHANNEL,
                                   OUT_CHANNEL],
                                  None,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=5e-2,
                                      dtype=DTYPE))
    
        for layer in S[1:]:
            s = statespace[layer]
            
            if s[0]=='Start':
                out = inputimage
            elif s[0] == 'C':
                out = conv(out,S)
            elif s[0] == 'P':
                out = pool(out,S)
            elif s[0] == 'FC':
                out = fullconnect(out,S)
                
        out = globalavgpool/softmax

            
            
if __name__ == '__main__':
    
    alpha = 0.01
    M = 3
    epsilon = 1
    K = 1
    
    sa = SA.getstate()
    NUMSTATES, S = sa.countstates()
    NUMACTIONS,A = sa.countactions()
    Q,Sprime,Aprime = Q_learning(NUMSTATES,NUMACTIONS,M,epsilon,K,S,A)
    
