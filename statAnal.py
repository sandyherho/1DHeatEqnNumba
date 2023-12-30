#!/usr/bin/env python

"""
statAnal.py

Find the statistical differences between
2 Total Runtime Experiments:
without Numba vs. Numba

SHSH <sandy.herho@email.ucr.edu>
30/12/23
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("bmh")


def process_data(file_path):
    # Read CSV file and extract control and test data
    df = pd.read_csv(file_path)
    print(df.describe().round(3))
    control = df["control"].to_numpy()
    test = df["test"].to_numpy()
    return control, test


def print_test_results(test_name, test_statistic, p_value):
    # Print results of statistical tests
    print(f"{test_name}:")
    print(f"Test statistic: {test_statistic}")
    print(f"P-value: {p_value}")
    if p_value < 0.01:
        print(f"The {test_name} indicates a significant difference between the control and test groups.")
    else:
        print(f"The {test_name} does not indicate a significant difference between the control and test groups.")
    print("\n")


def plot_and_save_histogram(control, test, filename):
    # Plot and save histogram
    sns.histplot(control, bins=15, kde=True, label='without Numba')
    sns.histplot(test, bins=15, kde=True, label='with Numba')
    plt.xlabel("Total Runtime [seconds]", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.legend()
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_and_save_boxplot(control, test, filename):
    # Plot and save boxplot
    sns.boxplot(x=['Control'] * len(control) + ['Test'] * len(test), y=np.concatenate([control, test]))
    plt.xticks([0, 1], ['without Numba', 'with Numba'])
    plt.ylabel("Total Runtime [seconds]", fontsize=16)
    plt.xlabel("Treatment", fontsize=16)
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_and_save_violinplot(control, test, filename):
    # Plot and save violinplot
    vio = pd.DataFrame({'Value': np.concatenate([control, test]),
                        'Treatment': ['without Numba'] * len(control) + ['with Numba'] * len(test)})

    sns.violinplot(x='Treatment', y='Value', data=vio, inner="quartile", hue='Treatment')
    plt.ylabel("Total Runtime [seconds]", fontsize=16)
    plt.xlabel("Treatment", fontsize=16)
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_and_save_ecdf(control, test, filename):
    # Plot and save empirical cumulative distribution functions (ECDFs)
    sorted_control = np.sort(control)
    y_control = np.arange(1, len(sorted_control) + 1) / len(sorted_control)
    plt.plot(sorted_control, y_control, marker='.', linestyle='--', label='without Numba')

    sorted_test = np.sort(test)
    y_test = np.arange(1, len(sorted_test) + 1) / len(sorted_test)
    plt.plot(sorted_test, y_test, marker='.', linestyle='--', label='with Numba')
    plt.xlabel("Total Runtime [seconds]", fontsize=16)
    plt.ylabel("Empirical CDFs", fontsize=16)
    plt.legend()
    plt.savefig(filename, dpi=400)
    plt.close()


if __name__ == "__main__":
    # Explicit Scheme
    explicit_control, explicit_test = process_data("../1DHeatNumba/runtime_data/explicit_scheme_runtime.csv")
    differences = explicit_control - explicit_test
    mannwhitney_result = mannwhitneyu(explicit_control, explicit_test)
    wilcoxon_result = wilcoxon(differences)
    print_test_results("Mann-Whitney U Test", mannwhitney_result.statistic, mannwhitney_result.pvalue)
    print_test_results("Wilcoxon Signed-Rank Test", wilcoxon_result.statistic, wilcoxon_result.pvalue)

    plot_and_save_histogram(explicit_control, explicit_test, "./figs/fig4a.png")
    plot_and_save_boxplot(explicit_control, explicit_test, "./figs/fig4b.png")
    plot_and_save_violinplot(explicit_control, explicit_test, "./figs/fig4c.png")
    plot_and_save_ecdf(explicit_control, explicit_test, "./figs/fig4d.png")

    # Implicit Scheme
    implicit_control, implicit_test = process_data("../1DHeatNumba/runtime_data/implicit_scheme_runtime.csv")
    differences = implicit_control - implicit_test
    mannwhitney_result = mannwhitneyu(implicit_control, implicit_test)
    wilcoxon_result = wilcoxon(differences)
    print_test_results("Mann-Whitney U Test", mannwhitney_result.statistic, mannwhitney_result.pvalue)
    print_test_results("Wilcoxon Signed-Rank Test", wilcoxon_result.statistic, wilcoxon_result.pvalue)

    plot_and_save_histogram(implicit_control, implicit_test, "./figs/fig5a.png")
    plot_and_save_boxplot(implicit_control, implicit_test, "./figs/fig5b.png")
    plot_and_save_violinplot(implicit_control, implicit_test, "./figs/fig5c.png")
    plot_and_save_ecdf(implicit_control, implicit_test, "./figs/fig5d.png")
