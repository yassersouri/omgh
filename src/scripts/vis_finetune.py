import numpy as np
import pandas as pd
import re
import click
from matplotlib import pylab as plt


@click.command()
@click.argument('log_file', type=click.Path(exists=True))
def main(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []

    for r in re.findall(accuracy_pattern, log):
        accuracy_iterations.append(int(r[0]))
        accuracies.append(float(r[1])*100)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)

    disp_results(loss_iterations, losses, accuracy_iterations, accuracies)


def disp_results(loss_iterations, losses, accuracy_iterations, accuracies):
    plt.plot(loss_iterations, losses)
    plt.plot(accuracy_iterations, accuracies)
    plt.show()


if __name__ == '__main__':
    main()
