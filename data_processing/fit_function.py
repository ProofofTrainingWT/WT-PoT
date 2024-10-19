import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.optimize import curve_fit

from config import opt
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from statistic_trajectory_watermark import read_csv_file, cal_area


def exponential_decay(x):
    return 1*np.exp(-x)

def power_function(x):
    return 1/x**2

def tanh_function(x):
    return np.tanh(x)

# def logistic_function(x):
#     return 1 / (1 + np.exp(-0.1*x-1))
#
# def logistic_decay_function(x):
#     return 1 / (1 + np.exp(0.1*x-6))

def logistic_function(x, a, b, c, d):
    return 65 / ((1 + np.exp(-a * x - b))*(1 + np.exp(c * x - d)))

if __name__ == '__main__':
    opt.logpath = '../follow_log/({}){}_{}_{}/'.format(8, opt.dataset,
                                                       opt.model_type, opt.follow_tag)  # output path
    overall_dataset = []
    for log_index in range(6, 7):
        opt.logname = osp.join(opt.logpath, 'log({})_ep10.tsv'.format(log_index))
        overall_dataset.append(read_csv_file(opt.logname))

    index = list(range(56, 1726, 10))  # 1726 1226
    reference_area = []
    reference_area_counterpart = []
    areas = []
    areas_counterpart = []
    for exp in overall_dataset:
        reference_area.append(cal_area(exp[40: 40 + 12])[0])
        reference_area_counterpart.append(cal_area(exp[40: 40 + 12])[1])
        traj_areas = []
        traj_areas_c = []
        for wm_id in index:
            traj_areas.append(cal_area(exp[wm_id: (wm_id + 6)])[0])
            traj_areas_c.append(cal_area(exp[wm_id: (wm_id + 6)])[1])
        areas.append(traj_areas)
        areas_counterpart.append(traj_areas_c)
    areas = np.asarray(areas)
    # areas_counterpart = np.asarray(areas_counterpart)
    x = np.linspace(0, len(areas[0]), num=len(areas[0]))
    # print(len(x))
    # print(len(areas[0]))
    popt, pcov = curve_fit(logistic_function, x, areas[0])
    #y = 50*logistic_decay_function(x)*logistic_function(x)#tanh_function(x)#power_function(x)#logistic_function(x)
    a_fit, b_fit, c_fit, d_fit= popt
    print(popt)

    plt.plot(x, logistic_function(x, a_fit, b_fit, c_fit, d_fit))
    plt.scatter(x, areas[0], marker='.')
    plt.show()