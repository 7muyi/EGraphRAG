import numpy as np


def get_cos_sim_matrix(x: np.array, y: np.array) -> np.array:
    # Cosine_similarity(x, y) = (xy) / (||x|| ||y||)
    x_norms = np.linalg.norm(x, axis=1, keepdims=True)
    y_norms = np.linalg.norm(y, axis=1, keepdims=True)
    cos_sim_matrix =  (x @ y.T) / (x_norms @ y_norms.T)
    return cos_sim_matrix

def retrieve(query: np.array, target: np.array, top_k: int = 1) -> int:
    cos_sim_matrix = get_cos_sim_matrix(query, target)
    
    if top_k == 1:
        max_value = np.max(cos_sim_matrix, axis=1)
        max_indices = np.argmax(cos_sim_matrix, axis=1)
        return max_value, max_indices
    else:
        # Index of top k element
        top_k_indicies = np.argsort(cos_sim_matrix, axis=1)[:, -top_k:][:, ::-1]
        # Value of top k element
        top_k_values = np.take_along_axis(cos_sim_matrix, top_k_indicies, axis=1)
        return top_k_values.tolist(), top_k_indicies.tolist()