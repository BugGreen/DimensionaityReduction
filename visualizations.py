import numpy as np
import matplotlib.pyplot as plt
from typing import List


def generate_points_3vars(eq: List[float], num_points: int = 100) -> np.ndarray:
    """
    Generates points that satisfy the given linear equation by randomly selecting values for two variables and solving
    for the third.

    :param eq: The coefficients of the equation [a, b, c, constant] corresponding to ax + by + cz = constant.
    :param num_points: Number of points to generate.
    :return: An array of points that satisfy the equation.
    """
    points = []
    np.random.seed(0)
    for _ in range(num_points):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        # Assuming the equation is always solvable for z
        if eq[2] != 0:  # cz = constant - ax - by -> z = (constant - ax - by) / c
            z = (eq[3] - eq[0]*x - eq[1]*y) / eq[2]
            points.append([x, y, z])
        else:  # Handle cases where c=0 and equation must be solved for another variable
            # This is a simplification and may need adjustment based on specific equations
            z = np.random.uniform(-10, 10)
            if eq[1] != 0:  # by = constant - ax - cz -> y = (constant - ax - cz) / b
                y = (eq[3] - eq[0]*x - eq[2]*z) / eq[1]
            points.append([x, y, z])
    return np.array(points)


def plot_reduced_data(reduced_data: np.ndarray, labels: List[str], reduction_technique: str = 'PCA'):
    """
    Plots the reduced data from dimensionality reduction techniques.

    :param reduced_data: A numpy array of reduced data points where rows represent equations.
    :param labels: A list of strings representing labels for the data points.
    :param reduction_technique: technique used for dimensionality reduction.
    :return: None.
    """
    plt.figure(figsize=(8, 6))
    # Distinguish between system equations and the given equation for coloring
    for i, label in enumerate(labels):
        if label == 'system':
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color='blue', label='System Equations' if
            'System Equations' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color='red', label='Given Equation' if
            'Given Equation' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title('2D Projection of Linear Equations after ' + reduction_technique)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()
