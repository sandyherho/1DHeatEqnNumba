#!/usr/bin/env python

"""
numba_1DHeatEqn.py

Fast 1D Heat Equation Solver

SHSH <sandy.herho@email.ucr.edu>
28/12/23
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

plt.style.use("bmh")

@jit(nopython=True, parallel=True)
def linspace(start, stop, num):
    step = (stop - start) / (num - 1)
    return np.linspace(start, stop, num)

@jit(nopython=True, parallel=True)
def full_like(arr, fill_value):
    return np.full_like(arr, fill_value)

@jit(nopython=True, parallel=True)
def calculate_heat_equation(delta_t, num_x, alpha, t_max, temp1, temp2, scheme):
    delta_x = 1.0 / (num_x - 1)
    C = alpha * delta_t / (delta_x * delta_x)

    x = linspace(0, 1, num_x)
    y = full_like(x, temp1)
    y[-1] = temp2

    time = 0
    count = 0
    num_time_steps = int(round(t_max / delta_t))
    pause_percentages = [1, 4, 10, 20, 100]
    pause_time_steps = [int(round(num_time_steps * 0.01 * p)) for p in pause_percentages]

    tri_diag = np.zeros((num_x - 2, num_x - 2))

    if scheme == "implicit":
        np.fill_diagonal(tri_diag, 1 + 2 * C)
        np.fill_diagonal(tri_diag[1:], -C)
        np.fill_diagonal(tri_diag[:, 1:], -C)

    data = []

    # Plot initial conditions
    data.append((x.copy(), y.copy(), "Initial Condition"))

    while time < t_max:
        y_old = y.copy()

        if scheme == "explicit":
            for i in prange(1, num_x - 1):
                y[i] = y_old[i] + C * (y_old[i + 1] - 2 * y_old[i] + y_old[i - 1])
        else:
            rhs = y_old[1:-1]
            rhs[0] += C * y_old[0]
            rhs[-1] += C * y_old[-1]
            for i in prange(num_x - 2):
                y[i + 1] = np.linalg.solve(tri_diag, rhs)[i]

        time += delta_t
        count += 1

        if count in pause_time_steps:
            index = pause_time_steps.index(count)
            # Create a new NumPy array explicitly
            data.append((x.copy(), y.copy(), f"{pause_percentages[index]}% of tMax"))

    return data

def plot_results(data):
    plt.figure()
    for x, y, label in data:
        plt.plot(x, y, "-", label=label, linewidth=3)

    plt.xlim(0, 1)
    plt.xticks(fontsize=14)
    plt.ylim(temp1, temp2)
    plt.yticks(fontsize=14)
    plt.ylabel("Temperature [$^{\circ}$C]", fontsize=18, loc="center", rotation=90)
    plt.xlabel("$x$ [m]", fontsize=18)
    plt.legend(prop={"size": 14})
    plt.savefig("fig.png", dpi=300)

if __name__ == "__main__":
    num_x = 101
    t_max = 10
    alpha = 0.2  # Plexiglass

    # Boundary Conditions
    temp1 = 20
    temp2 = 100

    # Implicit Scheme Settings
    delta_t = 0.1 # 0.1; 0.25; 2.5; 10
    scheme = "implicit"

    # Explicit Scheme Settings
    #delta_t = 0.000025
    #scheme = "explicit"

    data = calculate_heat_equation(delta_t, num_x, alpha, t_max, temp1, temp2, scheme)
    plot_results(data)
