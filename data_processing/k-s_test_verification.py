import ast
import csv
import os

from config import opt
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from wt_utils import cal_area

import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import t, norm, ks_2samp
import torch.nn as nn
import torch.optim as optim


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index > 1:
                data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)

def sigmoid(x, a1, a2, a3):
    # a3 = 1
    # a1 = 0.15
    return 100 / (1 + np.exp(- a1 * x + a2)) + 20 * a3

def nn_based_ks_test(x_hard, y_hard, y_hard_c, model, factor):
    print("using nn-based ks test[WT-POL]")
    plt.scatter(x_hard, y_hard)
    plt.scatter(x_hard, y_hard_c)
    predicted = model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()
    plt.plot(x_hard, predicted, label='Fitted Line')
    # plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_hard_trendline_nn.png')
    plt.close()

    y_hard = y_hard - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()
    y_hard_c = y_hard_c - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()

    # plt.scatter(x_hard, y_hard)
    # plt.scatter(x_hard, y_hard_c)
    # #plt.show()
    # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_detrend_nn.png')
    # plt.close()

    # construct the theoretical distribution # use
    def process_window(window):
        slope, intercept = np.polyfit(np.arange(len(window)), window, 1)
        trend = slope * np.arange(len(window)) + intercept
        detrended = window - trend
        zero_mean = detrended - detrended.mean()
        return zero_mean

    window_size = 10
    processed_data = []
    for i in range(0, len(y_hard_c), window_size):
        window = y_hard_c[i:i + window_size]
        processed_data.extend(process_window(window))

    processed_data = processed_data + np.random.normal(-0, 5, len(processed_data))
    y_hard = y_hard + np.random.normal(-0, 5, len(y_hard))
    y_hard_c = y_hard_c + np.random.normal(-0, 5, len(y_hard_c))

    plt.scatter(range(len(y_hard)), y_hard)
    plt.scatter(range(len(y_hard)), y_hard_c)
    # plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_detrend_nn.png')
    plt.close()

    # compare and produce 'confidence'
    statistic1, p_value1 = ks_2samp(processed_data, y_hard)
    # print(f"Benign_Statistic: {statistic1}")
    print(f"Benign P-value: {p_value1}")
    statistic, p_value = ks_2samp(processed_data, y_hard_c)
    # print(f"Abnormal Statistic: {statistic}")
    print(f"Abnormal P-value: {p_value}")

    with open(opt.result_logname, 'a+') as f:
        cols = ['nn_ks_test',
            "{}".format(statistic1),
                      "{}".format(p_value1),
                      "{}".format(statistic),
                      "{}".format(p_value)
                      ]
        f.write('\t'.join([str(c) for c in cols]) + '\n')

def fit_based_ks_test(x_hard, y_hard, y_hard_c, params, factor):
    plt.scatter(x_hard, y_hard)
    plt.scatter(x_hard, y_hard_c)
    predicted = sigmoid(((x_hard - opt.moving) / factor), *params)
    # plt.plot(x_hard, predicted, label='Fitted Line')
    # plt.show()
    # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_hard_trendline_fit.png')
    # plt.close()

    print("using fit-based ks test[WT-POL]")
    y_hard = y_hard - sigmoid(((x_hard - opt.moving) / factor), *params)
    y_hard_c = y_hard_c - sigmoid((x_hard - opt.moving) / factor, *params)
    # construct the theoretical distribution # use
    def process_window(window):
        slope, intercept = np.polyfit(np.arange(len(window)), window, 1)
        trend = slope * np.arange(len(window)) + intercept
        detrended = window - trend
        zero_mean = detrended - detrended.mean()
        return zero_mean

    window_size = 10
    processed_data = []

    for i in range(0, len(y_hard_c), window_size):
        window = y_hard_c[i:i + window_size]
        processed_data.extend(process_window(window))

    processed_data = processed_data + np.random.normal(-0, 5, len(processed_data))
    y_hard = y_hard + np.random.normal(-0, 5, len(y_hard))
    y_hard_c = y_hard_c + np.random.normal(-0, 5, len(y_hard_c))

    plt.scatter(range(len(y_hard)), y_hard)
    plt.scatter(range(len(y_hard)), y_hard_c)
    # plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_detrend_fit.png')
    plt.close()

    # compare and produce 'confidence'
    statistic1, p_value1 = ks_2samp(processed_data, y_hard)
    # print(f"Statistic: {statistic1}")
    print(f"P-value: {p_value1}")
    statistic, p_value = ks_2samp(processed_data, y_hard_c)
    # print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
    with open(opt.result_logname, 'a+') as f:
        cols = ['fit_ks_test',
            "{}".format(statistic1),
                      "{}".format(p_value1),
                      "{}".format(statistic),
                      "{}".format(p_value)
                      ]
        f.write('\t'.join([str(c) for c in cols]) + '\n')

def fit_based_ks_test_improved(x_hard, y_hard, y_hard_c, params, factor, y_hard_ref, params_ref, mode='ref1'):
    print("using fit-based ks test[WT-POL*]")
    #print("y_hard_ref", np.mean(y_hard_ref[:5]))
    #print("y_hard", np.mean(y_hard[:5]))

    trendline0 = sigmoid(((x_hard[0] - opt.moving) / factor), *params) - 1e-2
    trendline0_ref = sigmoid(x_hard[0] - 4 * opt.soft_point_num + 1, *params_ref) - 1e-2
    #print("trendline0", trendline0)
    y_hard_ref = (y_hard_ref - trendline0_ref) * ((sigmoid(((x_hard - opt.moving) / factor), *params) - trendline0) /
                                                  (sigmoid((x_hard-4 * opt.soft_point_num + 1), * params_ref)-trendline0_ref)) \
                 + trendline0 - sigmoid(((x_hard - opt.moving) / factor), *params)

    # print("y_hard_ref", y_hard_ref)
    #y_hard_ref = y_hard_ref - sigmoid(((x_hard - opt.moving) / factor), *params)

    y_hard = y_hard - sigmoid(((x_hard - opt.moving) / factor), *params)
    y_hard_c = y_hard_c - sigmoid(((x_hard - opt.moving) / factor), *params)

    plt.scatter(x_hard, y_hard_ref, label='Adjusted Reference Training')
    plt.xlabel('watemark round')
    plt.ylabel('detrended watermark signals')
    # plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_adjusted_nn_{mode}.png')
    plt.close()

    # construct the theoretical distribution # use
    def process_window_pivot_ref(window):
        slope, intercept = np.polyfit(np.arange(len(window)), window, 1)
        trend = slope * np.arange(len(window)) + intercept
        detrended = window - trend
        zero_mean = detrended - detrended.mean()
        return zero_mean

    def process_window(window, window_ref):
        slope, intercept = np.polyfit(np.arange(len(window_ref)), window_ref, 1)
        trend = slope * np.arange(len(window_ref)) + intercept
        detrended = window - trend
        # zero_mean = detrended - detrended.mean()
        return detrended

    window_size = 10
    processed_data = []
    processed_data_c = []
    y_hard_procssed = []
    y_hard_c_procssed = []
    for i in range(0, len(y_hard_c), window_size):
        window = y_hard_ref[i:i + window_size]
        window_ref = y_hard_ref[i:i + window_size]

        processed_data.extend(process_window(window, window_ref))
        window_c = y_hard_c[i:i + window_size]
        processed_data_c.extend(process_window_pivot_ref(window_c))

        window = y_hard[i:i + window_size]
        window_c = y_hard_c[i:i + window_size]
        y_hard_procssed.extend(process_window(window, window_ref))
        y_hard_c_procssed.extend(process_window(window_c, window_ref))

    processed_data = processed_data + np.random.normal(-5, 5, len(processed_data))
    y_hard_procssed = y_hard_procssed + np.random.normal(-5, 5, len(y_hard_procssed))
    y_hard_c_procssed = y_hard_c_procssed + np.random.normal(-5, 5, len(y_hard_c_procssed))

    plt.scatter(x_hard, y_hard_procssed, label='Benign Training')
    plt.scatter(x_hard, y_hard_c_procssed, label='Abnormal Training')
    plt.xlabel('watemark round')
    plt.ylabel('detrended watermark signals')
    # plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_detrend_imrpoved_fit_{mode}.png')
    plt.close()

    statistic1, p_value1 = ks_2samp(processed_data, y_hard_procssed)
    # print(f"Benign Statistic: {statistic1}")
    print(f"Benign P-value: {p_value1}")
    statistic, p_value = ks_2samp(processed_data, y_hard_c_procssed)
    # print(f"Abnormal Statistic: {statistic}")
    print(f"Abnormal P-value: {p_value}")

    with open(opt.result_logname, 'a+') as f:
        cols = ['fit_ks_test_improved',
            "{}".format(statistic1),
                      "{}".format(p_value1),
                      "{}".format(statistic),
                      "{}".format(p_value)
                      ]
        f.write('\t'.join([str(c) for c in cols]) + '\n')

def nn_based_ks_test_improved(x_hard, y_hard, y_hard_c, model, factor, y_hard_ref, model_ref, mode='ref1'):
    print("using nn-based ks test[WT-POL*]")

    trendline0 = model(torch.FloatTensor((x_hard[0:1] - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()[0] - 0.001
    trendline0_ref = model_ref(torch.FloatTensor(x_hard[0:1]).view(-1, 1)).view(-1).detach().numpy()[0] - 0.001
    y_hard_ref = (y_hard_ref - trendline0_ref) * ((model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy() - trendline0) /
                                                (model_ref(torch.FloatTensor(x_hard).view(-1, 1)).view(-1).detach().numpy() - trendline0_ref)) \
                 + trendline0 - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()

    y_hard = y_hard - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()
    y_hard_c = y_hard_c - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()
    #y_hard_ref = y_hard_ref - model(torch.FloatTensor((x_hard - opt.moving) / factor).view(-1, 1)).view(-1).detach().numpy()

    # construct the theoretical distribution # use
    def process_window_pivot_sample(window):
        slope, intercept = np.polyfit(np.arange(len(window)), window, 1)
        trend = slope * np.arange(len(window)) + intercept
        detrended = window - trend
        zero_mean = detrended - detrended.mean()
        return zero_mean

    def process_window(window, window_ref):
        slope, intercept = np.polyfit(np.arange(len(window_ref)), window_ref, 1)
        trend = slope * np.arange(len(window_ref)) + intercept
        detrended = window - trend
        return detrended

    window_size = 10
    processed_data = []
    processed_data_c = []
    y_hard_procssed = []
    y_hard_c_procssed = []
    for i in range(0, len(y_hard_c), window_size):
        window = y_hard_ref[i:i + window_size]
        window_ref = y_hard_ref[i:i + window_size]

        processed_data.extend(process_window(window, window_ref))
        window_c = y_hard_c[i: i + window_size]
        processed_data_c.extend(process_window_pivot_sample(window_c))

        window = y_hard[i:i + window_size]
        window_c = y_hard_c[i:i + window_size]
        y_hard_procssed.extend(process_window(window, window_ref))
        y_hard_c_procssed.extend(process_window(window_c, window_ref))

    # plt.scatter(x_hard, y_hard, label='Benign Training')
    # plt.scatter(x_hard, y_hard_c, label='Abnormal Training')
    # plt.xlabel('watemark round')
    # plt.ylabel('detrended watermark signals')
    #plt.show()
    # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_detrend_improved_nn_{mode}.png')
    # plt.close()

    processed_data = processed_data + np.random.normal(-0, 5, len(processed_data))
    y_hard_procssed = y_hard_procssed + np.random.normal(-0, 5, len(y_hard_procssed))
    y_hard_c_procssed = y_hard_c_procssed + np.random.normal(-0, 5, len(y_hard_c_procssed))

    statistic1, p_value1 = ks_2samp(processed_data, y_hard_procssed)
    # print(f"Benign Statistic: {statistic1}")
    print(f"Benign P-value: {p_value1}")
    statistic, p_value = ks_2samp(processed_data, y_hard_c_procssed)
    # print(f"Abnormal Statistic: {statistic}")
    print(f"Abnormal P-value: {p_value}")

    with open(opt.result_logname, 'a+') as f:
        cols = ['nn_ks_test_improved',
            "{}".format(statistic1),
                      "{}".format(p_value1),
                      "{}".format(statistic),
                      "{}".format(p_value)
                      ]
        f.write('\t'.join([str(c) for c in cols]) + '\n')

def piecewise_dual_linear(p, x):
    k1, k2, m1, m2, x0 = p
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1*x + m1, lambda x: k2*x + m2])

def single_linear(p, x):
    k, m = p
    return k*x + m

def main():
    opt.logpath = '.' + opt.logpath
    overall_dataset = []
    print("opt.log_name", opt.log_name)
    opt.log_name = ast.literal_eval(opt.log_name)

    reference_index = 0
    plot_index = 1

    for suffix in opt.log_name:
        opt.logname = osp.join(opt.logpath, 'log_{}-{}_{}.tsv'.format(opt.wm_num, opt.cl_num, suffix))
        overall_dataset.append(read_csv_file(opt.logname))

    if not osp.exists(osp.join(opt.logpath, 'results')):
        os.makedirs(osp.join(opt.logpath, 'results'))

    opt.result_logname = osp.join(opt.logpath, 'results', 'process_{}-{}_{}.tsv'.format(opt.wm_num,
                                                                        opt.cl_num, opt.log_name[plot_index]
                                                                        ))
    if not osp.exists(osp.join(opt.logpath, 'results')):
        os.makedirs(osp.join(opt.logpath, 'results'))

    with open(opt.result_logname, 'w+') as f:
        columns = ['Methods_Name',
                   'Benign_Statistic',
                   'Benign_p_value',
                   'Abnormal_Statistic',
                   'Abnormal_p_value'
                   ]

        f.write('\t'.join(columns) + '\n')

    clean_loss = []
    wm_loss = []
    clean_acc = []
    benign_training = []
    wm_round = []
    for exp in overall_dataset:
        clean_loss.append(exp[:, 1])
        clean_acc.append(exp[:, 2])
        wm_loss.append(exp[:, 3])
        benign_training.append(exp[:, 4])
        wm_round.append(exp[:, 0])

    x = list(range(len(clean_loss[plot_index][250:])))

    # area
    interval_points_soft = 6
    interval_points_hard = 6
    new_mechanism_process = 41
    hard_new_mec_process = 109
    start_index = 170   #170for gtsrb(21-24)  # (9) #0 (8)# 207 (6)
    cleanse_points_soft = 36#30  # 40 (3)
    cleanse_points_hard = 36#30  # 40+14 (3)

    areas = []
    areas_counterpart = []
    for id, exp in enumerate(overall_dataset):
        traj_areas = []
        traj_areas_c = []
        epoch_index = list(
            range(start_index, start_index + 4 * ((interval_points_soft + cleanse_points_soft) * opt.soft_point_num +
                                                  new_mechanism_process),
                  (interval_points_soft + cleanse_points_soft) * opt.soft_point_num + new_mechanism_process))
        index = []
        for e_id in epoch_index:
            index += list(range(e_id, e_id + (interval_points_soft + cleanse_points_soft) * opt.soft_point_num,
                                (interval_points_soft + cleanse_points_soft)))
            print("len(index)", len(index))

        for wm_id in index:
            print("index", wm_id)
            traj_areas.append(cal_area(exp[(wm_id): (wm_id + interval_points_soft)], index=[4, 6])[0])
            traj_areas_c.append(cal_area(exp[(wm_id): (wm_id + interval_points_soft)], index=[4, 6])[1])

        hard_start_index = start_index + 4 * (
                (interval_points_soft + cleanse_points_soft) * opt.soft_point_num + new_mechanism_process)
        #overall epoch=10
        epoch_index = list(range(hard_start_index, hard_start_index + 12 * (
                (interval_points_hard + cleanse_points_hard) * opt.hard_point_num + hard_new_mec_process),
                                 (interval_points_hard + cleanse_points_hard) * opt.hard_point_num + hard_new_mec_process))

        index = []
        for e_id in epoch_index:
            index += list(range(e_id, e_id + (interval_points_hard + cleanse_points_hard) * opt.hard_point_num,
                                (interval_points_hard + cleanse_points_hard)))
            print("len(index)", len(index))

        for wm_id in index:
            print("hard index", wm_id)
            try:
                if wm_id in epoch_index:
                    traj_areas.append(cal_area(exp[(wm_id): (wm_id + interval_points_hard)], index=[4, 6])[0])
                else:
                    traj_areas.append(cal_area(exp[(wm_id): (wm_id + interval_points_hard)], index=[4, 6])[0])
            except:
                print("zero_index", wm_id)

            if wm_id in epoch_index:
                traj_areas_c.append(cal_area(exp[(wm_id): (wm_id + interval_points_hard)], index=[4, 6])[1])
            else:
                traj_areas_c.append(cal_area(exp[(wm_id): (wm_id + interval_points_hard)], index=[4, 6])[1])
        areas.append(traj_areas)
        areas_counterpart.append(traj_areas_c)

    x = np.linspace(0, 0 + len(areas[plot_index]), num=len(areas[plot_index]))
    plt.scatter(x[4*opt.soft_point_num:], areas[plot_index][4*opt.soft_point_num:], marker='.', label='benign-' + opt.log_name[plot_index])
    plt.scatter((x[:4*opt.soft_point_num] * opt.factor + opt.moving), areas[plot_index][:4*opt.soft_point_num],
                marker='.', label='pivot-benign-' + opt.log_name[plot_index])

    plt.scatter(x[4*opt.soft_point_num:], areas_counterpart[plot_index][4*opt.soft_point_num:],
                marker='.', label='pivot-abnormal-' + opt.log_name[plot_index])
    plt.xlabel('Watermark Round')
    plt.ylabel('Watermark Signal')
    plt.legend()
    #plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[plot_index]}_wm.png')
    plt.close()

    # trend line
    x_tensor = torch.FloatTensor(x[:4*opt.soft_point_num]).view(-1, 1)
    y_tensor = torch.FloatTensor(areas[plot_index][:4*opt.soft_point_num]).view(-1, 1)
    # fit trend
    params, covariance = curve_fit(sigmoid, x[:4*opt.soft_point_num], areas[plot_index][:4*opt.soft_point_num], maxfev=10000)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, 100)
            self.fc2 = nn.Linear(100, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 5000
    for epoch in range(epochs):
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    x_test_tensor = torch.FloatTensor(np.linspace(0, 4*opt.soft_point_num, num=50)).view(-1, 1)
    predicted = model(x_test_tensor).detach().numpy()
    y_fit = sigmoid(x_test_tensor.numpy(), *params)

    # plot trend line
    plt.scatter(x[:4*opt.soft_point_num], areas[plot_index][:4*opt.soft_point_num], label='Original Data')
    plt.plot(x_test_tensor, predicted, label='Fitted Line')
    plt.plot(x_test_tensor, y_fit, label='Sigmoid')
    plt.legend()
    #plt.show()
    plt.savefig(opt.logpath + f'\\results\\{opt.log_name[plot_index]}_fit.png')
    plt.close()

    # remove trend line; hard process is
    len_x = len(x[4*opt.soft_point_num:])
    x_hard = x[4*opt.soft_point_num:]
    y_hard = areas[plot_index][4*opt.soft_point_num:]
    y_hard_c = areas_counterpart[plot_index][4*opt.soft_point_num:]
    y_hard_ref = areas[reference_index][4*opt.soft_point_num:]
    # y_hard_ref2 = areas[reference_index2][4*opt.soft_point_num:]

    for i in range(len(y_hard)):
        if i % 4 == 0:
            if i == 0:
                y_hard[i] = y_hard[i + 1]
                y_hard_c[i] = y_hard_c[i + 1]
                y_hard_ref[i] = y_hard_ref[i + 1]
                # y_hard_ref2[i] = y_hard_ref2[i + 1]
            else:
                y_hard[i] = (y_hard[i - 1] + y_hard[i + 1]) / 2
                y_hard_c[i] = (y_hard_c[i - 1] + y_hard_c[i + 1]) / 2
                y_hard_ref[i] = (y_hard_ref[i - 1] + y_hard_ref[i + 1]) / 2
                # y_hard_ref2[i] = (y_hard_ref2[i - 1] + y_hard_ref2[i + 1]) / 2

    def objective_benign_double(p):
        return np.sum((y_hard - piecewise_dual_linear(p, x[4*opt.soft_point_num:]-4*opt.soft_point_num+1)) ** 2)

    def objective_benign_single(p):
        return np.sum((y_hard - single_linear(p, x[4*opt.soft_point_num:]-4*opt.soft_point_num+1)) ** 2)

    p_opt_double = minimize(objective_benign_double, [1, 1, 20, 20, 8], method='Nelder-Mead').x
    p_opt_single = minimize(objective_benign_single, [1, 1], method='Nelder-Mead').x
    aic_double = objective_benign_double(p_opt_double) #2 * 5 + 2 *
    aic_single = objective_benign_single(p_opt_single) #2 * 2 + 2 *

    if aic_single < aic_double:
        p_opt = p_opt_single
        is_continuous_benign = True
        y2_start_benign = y1_end_benign = 0

        plt.scatter(x[4*opt.soft_point_num:] - 4*opt.soft_point_num+1, y_hard)
        x_temp = x[4*opt.soft_point_num:] - 4*opt.soft_point_num+1
        # plt.plot(x_temp, single_linear(p_opt, x_temp))
        # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_piecewise_fit.png')
        # plt.close()
    else:
        p_opt = p_opt_double
        y1_end_benign = p_opt[0] * p_opt[4] + p_opt[2]
        y2_start_benign = p_opt[1] * p_opt[4] + p_opt[3]
        is_continuous_benign = np.isclose(y1_end_benign, y2_start_benign, atol=15)

        plt.scatter(x[4*opt.soft_point_num:] - 4*opt.soft_point_num+1, y_hard)
        x_temp = x[4*opt.soft_point_num:] - 4*opt.soft_point_num +1
        # plt.plot(x_temp[x_temp < p_opt[4]], piecewise_dual_linear(p_opt, x_temp[x_temp < p_opt[4]]))
        # plt.plot(x_temp[x_temp >= p_opt[4]], piecewise_dual_linear(p_opt, x_temp[x_temp >= p_opt[4]]))
        # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_piecewise_fit.png')
        # plt.close()

    def objective_abnormal_double(p):
        return np.sum((y_hard_c - piecewise_dual_linear(p, x[4*opt.soft_point_num:] - 4*opt.soft_point_num + 1)) ** 2)

    def objective_abnormal_single(p):
        return np.sum((y_hard_c - single_linear(p, x[4*opt.soft_point_num:] - 4*opt.soft_point_num + 1)) ** 2)

    p_opt_double = minimize(objective_abnormal_double, [1, 1, 20, 20, 8], method='Nelder-Mead').x
    p_opt_single = minimize(objective_abnormal_single, [1, 1], method='Nelder-Mead').x
    aic_double = objective_abnormal_double(p_opt_double) #2 * 5 + 2 *
    aic_single = objective_abnormal_single(p_opt_single) # 2 * 2 + 2 *
    print("(abnormal) double_loss", aic_double)
    print("(abnormal) single_loss", aic_single)
    if aic_single < aic_double:
        p_opt = p_opt_single
        is_continuous = True
        y2_start = y1_end = 0

        plt.scatter(x[4*opt.soft_point_num:] - 4*opt.soft_point_num +1, y_hard_c)
        x_temp = x[4*opt.soft_point_num:] - 4*opt.soft_point_num +1
        # plt.plot(x_temp, single_linear(p_opt, x_temp))
        # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_piecewise_fit_abnormal.png')
        # plt.close()
    else:
        p_opt = p_opt_double
        y1_end = p_opt[0] * p_opt[4] + p_opt[2]
        y2_start = p_opt[1] * p_opt[4] + p_opt[3]
        is_continuous = np.isclose(y1_end, y2_start, atol=15)

        plt.scatter(x[4*opt.soft_point_num:] - 4*opt.soft_point_num +1, y_hard_c)
        x_temp = x[4*opt.soft_point_num:] - 4*opt.soft_point_num +1
        # plt.plot(x_temp[x_temp < p_opt[4]], piecewise_dual_linear(p_opt, x_temp[x_temp < p_opt[4]]))
        # plt.plot(x_temp[x_temp >= p_opt[4]], piecewise_dual_linear(p_opt, x_temp[x_temp >= p_opt[4]]))
        # plt.savefig(opt.logpath + f'\\results\\{opt.log_name[1]}_piecewise_fit_abnormal.png')
        # plt.close()

    with open(opt.result_logname, 'a+') as f:
        cols = ['is_countinuous',
            "{}".format(is_continuous_benign),
            y2_start_benign-y1_end_benign,
            "{}".format(is_continuous),
            y2_start - y1_end
                ]
        f.write('\t'.join([str(c) for c in cols]) + '\n')

    print("WT-PoL")
    nn_based_ks_test(x_hard, y_hard, y_hard_c, model, opt.factor)
    fit_based_ks_test(x_hard, y_hard, y_hard_c, params, opt.factor)

    print("WT-PoL*")
    # trend line
    x_tensor_ref = torch.FloatTensor(x[4*opt.soft_point_num:]).view(-1, 1)
    y_tensor_ref = torch.FloatTensor(areas[reference_index][4*opt.soft_point_num:]).view(-1, 1)

    # fit trend
    params_ref, _ = curve_fit(sigmoid, x[4*opt.soft_point_num:]-4*opt.soft_point_num+1, areas[reference_index][4*opt.soft_point_num:])
    model_ref = Net()
    criterion = nn.MSELoss()
    optimizer_ref = optim.Adam(model_ref.parameters(), lr=0.01)
    epochs = 5000
    for epoch in range(epochs):
        outputs = model_ref(x_tensor_ref)
        loss = criterion(outputs, y_tensor_ref)
        optimizer_ref.zero_grad()
        loss.backward()
        optimizer_ref.step()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    nn_based_ks_test_improved(x_hard,
                              y_hard,
                              y_hard_c,
                              model, opt.factor, y_hard_ref, model_ref)
    fit_based_ks_test_improved(x_hard,
                               y_hard,
                               y_hard_c,
                               params, opt.factor, y_hard_ref, params_ref)

if __name__ == '__main__':
    main()
