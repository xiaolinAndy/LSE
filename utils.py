import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# getting y_i given x1, x2, noise = [mu, standard_sigma]
def data_acquire(x1, x2, noise):
    y = np.sin(10*x1) + np.cos(4*x2) + np.cos(3*x1*x2)
    if noise[1] != 0:
        y += np.random.normal(loc=noise[0], scale=noise[1], size=x1.shape)
    return y

# draw contour plot for data function
def draw_plot():
    delta = 0.01
    x = np.arange(0, 1, delta)
    y = np.arange(0, 2, delta)
    # X, Y are [100, 200]
    X, Y = np.meshgrid(x, y)
    Z, = data_acquire(X, Y, np.array([0, 0]))
    fig, ax = plt.subplots()
    # the fourth parameter is the contour line value
    CS = ax.contour(X, Y, Z, [-1, 0, 1, 2])
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Data distribution')
    plt.show()

# draw a simple F1 with step num plot
def draw_F1(F1_list):
    x = np.arange(0, len(F1_list))
    y = F1_list
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('F1 score')
    plt.show()

# draw F1 and step plots for multiple algos with single iterations
# F1_list:[[algo1: 0.05, 0.07, 0.1], [algo2:], ...]
def draw_F1s(F1_lists):
    max_len = max([len(l) for l in F1_lists])
    fig, ax = plt.subplots()
    for i, l in enumerate(F1_lists):
        # fill up to the same size
        x = np.arange(0, max_len)
        y = l + [l[-1]] * (max_len - len(l))
        ax.plot(x, y)
    ax.set_title('F1 scores')
    ax.set_xlabel('step')
    ax.set_ylabel('F1 score')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("images/f1_plt.png")

# draw F1 and cost plots for multiple algos with multiple iterations
# cost_list:[[algo1: 5, 7, 10], [algo2:], ...]
# F1_list:[[algo1: 0.05, 0.07, 0.1], [algo2:], ...]
# labels: [algo1.name, algo2.name, ...]
def draw_costs(cost_lists, F1_lists, labels):
    max_cost = max([l[-1] for l in cost_lists])
    fig, ax = plt.subplots()
    for i, l in enumerate(cost_lists):
        # fill up to the same size
        x = l + [max_cost]
        y = F1_lists[i] + [F1_lists[i][-1]]
        ax.plot(x, y, label=labels[i])
    ax.set_title('F1 vs cost')
    ax.set_xlabel('cost')
    ax.set_ylabel('F1 score')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("images/f1_cost.png")

# draw F1 and step plots for multiple algos with multiple iterations
# F1_lists looks like: [[algo1's data, such as:[iter1's data: 0.1, 0.15],[0.05, 0.2],[0.1, 0.15],...], [algo2's data]]
def f1_plots(F1_lists, labels):
    max_len = 100
    fig, ax = plt.subplots()
    fmt = ['o', 'x', '^']
    for i, all_f1 in enumerate(F1_lists):
        new_all_f1 = []
        for l in all_f1:
            l = l + [l[-1]] * (max_len - len(l))
            new_all_f1.append(l)
        boxes = []
        for k in range(0, max_len):
            boxes.append([new_all_f1[j][k] for j in range(len(new_all_f1))])
        box_array = np.array(boxes)
        mean = np.mean(box_array, axis=1)
        std_deviation = np.std(box_array, axis=1)
        plt.errorbar(range(max_len), mean, yerr=std_deviation, fmt=fmt[i], label=labels[i])
    ax.set_title('F1 scores')
    ax.set_xlabel('step')
    ax.set_ylabel('F1 score')
    plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("images/f1_box.png")

# draw picked points for a given algo
def draw_points(points, label, cost):
    delta = 0.01
    x = np.arange(0, 1, delta)
    y = np.arange(0, 2, delta)
    # X, Y are [100, 200]
    X, Y = np.meshgrid(x, y)
    Z = data_acquire(X, Y, np.array([0, 0]))
    fig, ax = plt.subplots()
    # the fourth parameter is the contour line value
    CS = ax.contour(X, Y, Z, [-1, 0, 1, 2])
    ax.clabel(CS, inline=1, fontsize=10)
    ax.scatter(points[:, 0], points[:, 1], color='red',marker='o')
    ax.set_title('Picked Points of ' + label)
    plt.show()
    plt.savefig('images/' + label + '_' + cost + 'points.png')

# draw paths of picked points for a given algo
def draw_paths(points, label, cost):
    delta = 0.01
    x = np.arange(0, 1, delta)
    y = np.arange(0, 2, delta)
    # X, Y are [100, 200]
    X, Y = np.meshgrid(x, y)
    Z = data_acquire(X, Y, np.array([0, 0]))
    fig, ax = plt.subplots()
    # the fourth parameter is the contour line value
    CS = ax.contour(X, Y, Z, [-1, 0, 1, 2])
    ax.clabel(CS, inline=1, fontsize=10)
    ax.plot(points[:, 0], points[:, 1], color='red')
    ax.set_title('Picked Point Paths of ' + label)
    plt.show()
    plt.savefig('images/' + label + '_' + cost + 'paths.png')


if __name__ == '__main__':
    draw_plot()