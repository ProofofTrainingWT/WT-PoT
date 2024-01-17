import csv
from config import opt
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index>1:
               data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)

def discrete_integral(x):
    integral = 0.0
    n = len(x)
    for i in range(n-1):
        w = (x[i+1] + x[i])/2
        integral += w
    return integral/n

def cal_area(data_array, index=[4, 6]):
    #area = discrete_integral(data_array[1:,1])#sum(data_array[1:,1])/len(data_array[1:, 1])
    #area_c = discrete_integral(data_array[1:, 2])#sum(data_array[1:, 2]) / len(data_array[1:, 2])
    start_index = 0
    end_index = 6
    # print(len(data_array[:, 1]))
    area = sum(data_array[:, index[0]])/len(data_array[:, index[0]])
    area_c =sum(data_array[:, index[1]])/len(data_array[:, index[1]])
    return area, area_c

def cal_max(data_array):
    #area = discrete_integral(data_array[1:,1])#sum(data_array[1:,1])/len(data_array[1:, 1])
    #area_c = discrete_integral(data_array[1:, 2])#sum(data_array[1:, 2]) / len(data_array[1:, 2])
    max1 = max(data_array[:, 1])
    max1_c =max(data_array[:, 2])
    return max1, max1_c
