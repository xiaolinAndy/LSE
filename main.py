from algos import Algo, LSE, LSE_imp, LSE_imp_mod, TRUVAR, TRUVAR_imp, RMILE
from argparse import ArgumentParser
from utils import f1_plots, draw_points, draw_paths, draw_costs
import random
import numpy as np

# return the average of a given list
def avg_list(l):
    sum = 0
    for value in l:
        sum += value
    avg = sum / len(l)
    return avg

# input parameters
def get_args():
    parser = ArgumentParser(description='High Level Set Estimation')
    # normal for F1 scores with steps, cost for F1 with costs, single for picked points and paths
    parser.add_argument('--test_type', type=str, default='normal')
    # algo num, 1 for LSE, ...see in README.txt
    parser.add_argument('--algo', type=int, nargs='+')
    # True for algos considering cost, false for not considering cost
    parser.add_argument('--cost', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    # set random seed
    random.seed(1)
    config = get_args()
    
    # generate sample data
    delta = 0.05
    x = np.arange(0, 1, delta)
    y = np.arange(0, 2, delta)
    X, Y = np.meshgrid(x, y)
    data = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    # [mu, sigma, l]
    GP_prior = [0, np.exp(1), np.exp(-1.5)]
    # different algos
    algo = {}
    algo[1] = LSE(data, GP_prior, 1, False, config.cost, acc=0)
    algo[2] = LSE_imp(data, GP_prior, 1 / 3, True, config.cost, acc=0)
    algo[3] = LSE_imp_mod(data, GP_prior, 1 / 3, True, config.cost, acc=0)
    algo[4] = TRUVAR(data, GP_prior, 1, False, config.cost, delta=0, eta=1, r=0.1)
    algo[5] = TRUVAR_imp(data, GP_prior, 1 / 3, False, config.cost, delta=0, eta=1, r=0.1)
    algo[6] = RMILE(data, GP_prior, 1, False, config.cost, eta=0.01)
    algo[7] = LSE(data, GP_prior, 1, False, True, acc=0)
    algo[7].name = 'LSE_cost'

    if config.test_type == 'single':
        # draw points and paths
        start_point = random.randint(0, data.shape[0])
        f1, cost, time, points = algo[config.algo[0]].run(start_point)
        cost = 'cost_' if config.cost else ''
        draw_points(points, algo[config.algo[0]].name, cost)
        draw_paths(points, algo[config.algo[0]].name, cost)
    elif config.test_type == 'cost':
        # draw cost and F1 plots
        f1s = []
        costs = []
        labels = []
        for j in config.algo:
            labels.append(algo[j].name)
        start_point = random.randint(0, data.shape[0])
        for index, j in enumerate(config.algo):
            print('Algo: ', algo[j].name)
            f1, cost, time, _ = algo[j].run(start_point)
            f1s.append(f1)
            costs.append(cost)
        draw_costs(costs, f1s, labels)
    else:
        # draw step and F1 plots
        f1s = []
        labels = []
        times = []
        for j in config.algo:
            f1s.append([])
            times.append([])
            labels.append(algo[j].name)
        # iteration steps, here we choose 10
        for i in range(10):
            print('Epoch: ', i)
            start_point = random.randint(0, data.shape[0])
            for index, j in enumerate(config.algo):
                print('Algo: ', algo[j].name)
                f1, cost, time, _ = algo[j].run(start_point)
                f1s[index].append(f1)
                times[index].append(time)
        for index, j in enumerate(config.algo):
            avg_time = avg_list(times[index])
            print(algo[j].name, 'avg_time: ', avg_time)
        f1_plots(f1s, labels)

if __name__ == '__main__':
    main()
