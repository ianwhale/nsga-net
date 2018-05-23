# -*- coding: utf-8 -*-
"""
Created on Fri May 04 15:26:56 2018

@author: yashe
"""

import numpy as np


def duplicate_check(encoding_1, encoding_2, n_nodes):
    # ...doing duplicate check by matchign connectivity-matrices...

    # ..compute connectivity matrices for each encoding
    M_1 = list_to_matrix(encoding_1, n_nodes)
    M_2 = list_to_matrix(encoding_2, n_nodes)

    duplicates = matching_arthitectures(M_1, M_2)

    return duplicates

def list_to_matrix(encoding_1, n_nodes):
    # ..finds the connectivity matrix for encoding_1 = [1,1,0,0,1,1,0,0,1,0,0,1,1,0,1]

    encoded_list_1 = [[] for i in range(n_nodes - 1)]

    counter = 0
    for i in range(n_nodes - 1):
        for j in range(i + 1):
            encoded_list_1[i].append(encoding_1[counter])
            counter = counter + 1

    # ..constructing a connectivity matrix..
    M = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes - 1):
        for j in range(i + 1):
            M[i + 1][j] = encoded_list_1[i][j]

    M_t = np.transpose(M)

    M_final = M_t - M

    return M_final


def sequence_pair(M):
    # ..determines input-output sequence...
    n_nodes = int(np.shape(M)[0])
    In_array = np.zeros(n_nodes)
    Out_array = np.zeros(n_nodes)

    pair_array = np.zeros([n_nodes, 2])
    for i in range(n_nodes):
        for j in range(n_nodes):
            if M[i][j] == -1:
                In_array[i] = In_array[i] + 1
            if M[i][j] == 1:
                Out_array[i] = Out_array[i] + M[i][j]
        pair_array[i][0] = int(In_array[i])
        pair_array[i][1] = int(Out_array[i])

    return pair_array


def sym_swap(M, i, j):
    n_nodes = int(np.shape(M)[0])
    M_temp1 = np.copy(M)

    M_temp1[i] = np.copy(M[j])
    M_temp1[j] = np.copy(M[i])

    M_temp2 = np.copy(M_temp1)

    for p in range(n_nodes):
        M_temp2[p][i] = M_temp1[p][j]
        M_temp2[p][j] = M_temp1[p][i]
    # ..swapping rows..

    return M_temp2


def matching_arthitectures(M_f, M_c):
    n_nodes = int(np.shape(M_f)[0])

    # ...Phase I... matching pairs....
    Pair_array_f = sequence_pair(M_f)
    Pair_array_c = sequence_pair(M_c)

    for i in range(n_nodes - 1):
        if np.array_equal(Pair_array_f, Pair_array_c):
            break

        if np.array_equal(Pair_array_c[i], Pair_array_f[i]):
            continue

        for j in range(i + 1, n_nodes):
            if np.array_equal(Pair_array_f[i], Pair_array_c[j]):
                k = j
                M_c = sym_swap(M_c, i, k)
                Pair_array_c = sequence_pair(M_c)

    if not (np.array_equal(Pair_array_c, Pair_array_f)):
        # print('phase 1 match successful')
        return False

    # ...Phase 2  match begins...

    for i in range(n_nodes - 1):
        if np.array_equal(M_f, M_c):
            break

        if np.array_equal(M_f[i], M_c[i]):
            continue

        for j in range(i + 1, n_nodes):
            if np.array_equal(M_f[i], M_c[j]):
                M_c = sym_swap(M_c, i, j)
                break

    if np.array_equal(M_f, M_c):
        return True
    else:
        return False
