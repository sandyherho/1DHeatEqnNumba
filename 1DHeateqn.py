#!/usr/bin/env python

"""
1D Heat Equation standard

SHSH <sandy.herho@ucr.edu>
12/27/23
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

def solve_heat_equation(delta_t, num_x, alpha, t_max, temp1, temp2, scheme):
    delta_x = 1.0 / (num_x - 1)
    C = alpha * delta_t / (delta_x * delta_x)

    x = np.linspace(0, 1, num_x)
    y = np.full_like(x, temp1)
    y[-1] = temp2

    plt.plot(x, y, "-", label="Initial Condition", linewidth=3)

    time = 0
    count = 0
    num_time_steps = int(np.rint(t_max / delta_t))
    pause_percentages = np.array([1, 4, 10, 20, 100])
    pause_time_steps = np.rint(num_time_steps * 0.01 * pause_percentages).astype(int)

    tri_diag = np.zeros((num_x - 2, num_x - 2))

    if scheme == "implicit":
        np.fill_diagonal(tri_diag, 1 + 2 * C)
        np.fill_diagonal(tri_diag[1:], -C)
        np.fill_diagonal(tri_diag[:, 1:], -C)

    while time < t_max:
        y_old = np.copy(y)

        if scheme == "explicit":
            y[1:-1] = y_old[1:-1] + C * (y_old[2:] - 2 * y_old[1:-1] + y_old[:-2])
        else:
            rhs = y_old[1:-1]
            rhs[0] += C * y_old[0]
            rhs[-1] += C * y_old[-1]
            y[1:-1] = np.linalg.solve(tri_diag, rhs)

        time += delta_t
        count += 1

        if count in pause_time_steps:
            index = np.where(pause_time_steps == count)
            plt.plot(x, y, "-", label=f"{pause_percentages[index][0]}% of tMax", linewidth=3)

    #plt.title("Temperature Distribution across Time", fontsize=24)
    plt.xlim(0, 1)
    plt.xticks(fontsize=14)
    plt.ylim(temp1, temp2)
    plt.yticks(fontsize=14)
    plt.ylabel("Temperature [$^{\circ}$C]", fontsize=18, loc="center", rotation=90)
    plt.xlabel("$x$ [m]", fontsize=18)
    plt.legend(prop={"size": 14})
    plt.savefig("fig1.png", dpi=300)
    #plt.show(block=True)

if __name__ == "__main__":
    num_x = 101
    t_max = 10
    alpha = 0.2  # Plexiglass

    # Boundary Conditions
    temp1 = 20
    temp2 = 100

    # Implicit Scheme Settings
    #delta_t = 10
    #scheme = "implicit"

    # Explicit Scheme Settings
    delta_t = 0.0001
    scheme = "explicit"

    solve_heat_equation(delta_t, num_x, alpha, t_max, temp1, temp2, scheme)
