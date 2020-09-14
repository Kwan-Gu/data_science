# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 00:08:56 2020

@author: WHITE
"""

#n = int(input())

table = [[1]*10]
for j in range(9): # 0, 8
    temp = [0]*(j+1)
    for i in range(j, 9):
        temp += [temp[-1]+table[j][i]]
    table.append(temp)

n = int(input())
sum = 0
for j in range(len(table)):
    for i in range(j,len(table[0])):
        sum+=table[j][i]
        if n>=sum:
            a, b = j, i
            break