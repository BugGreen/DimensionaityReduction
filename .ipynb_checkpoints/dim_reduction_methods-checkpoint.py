import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap.umap_


def apply_dimensionality_reduction(data: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
    """
    Applies specified dimensionality reduction technique to the given data.

    :param data: A numpy array where rows represent equations and columns represent coefficients and constants.
    :param method: The method of dimensionality reduction to apply ('pca', 'tsne', or 'umap').
    :param n_components: The number of dimensions to reduce the data to.
    :return: A numpy array containing the reduced data.
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Unsupported dimensionality reduction method: " + method)

    reduced_data = reducer.fit_transform(data)
    return reduced_data


def is_linear_combination(given_eq: list, system_eqs: list, constants: list, given_const: float) -> bool:
    """
    Checks if a given equation can be expressed as a linear combination of a system of equations,
    including the constant terms.

    :param given_eq: Coefficients of the given equation as a list [a1, a2, ..., an].
    :param system_eqs: List of lists, where each sublist represents coefficients of a system equation.
    :param constants: Constants of the system equations as a list [b1, b2, ..., bn].
    :param given_const: Constant term of the given equation.
    :return: True if the given equation is a linear combination of the system, False otherwise.
    """
    # Augment the system with the given equation
    A = np.array(system_eqs)
    b = np.array(constants)  # Existing constants of the system
    new_row = np.array(given_eq + [given_const])

    # Form the augmented matrix including constants
    augmented_matrix = np.vstack((np.column_stack((A, b)), new_row))

    # Perform row reduction
    rref = np.linalg.matrix_rank(np.column_stack((A, b)))
    rref_augmented = np.linalg.matrix_rank(augmented_matrix)

    # If ranks are the same, the given equation is a linear combination of the system
    return rref == rref_augmented
