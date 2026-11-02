
import numpy as np


def cosine_similarity(vector_a, vector_b):

    # Step 1: Convert inputs into NumPy arrays
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    # Step 2: Calculate dot product
    dot_product = np.dot(vector_a, vector_b)

    # Step 3: Calculate vector magnitudes
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # Step 4: Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cannot compare zero vectors")

    # Step 5: Calculate cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    return float(similarity)

def dot_product(vector_a, vector_b):

    import numpy as np

    # Convert inputs into NumPy arrays
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    # Check dimensions
    if vector_a.shape != vector_b.shape:
        raise ValueError("Both vectors must have the same dimensions")

    # Calculate dot product
    result = np.dot(vector_a, vector_b)

    return float(result)

def euclidean_distance(vector_a, vector_b):

    import numpy as np

    # Step 1: Convert inputs into NumPy arrays
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    # Step 2: Check vector dimensions
    if vector_a.shape != vector_b.shape:
        raise ValueError("Both vectors must have the same dimensions")

    # Step 3: Calculate differences
    difference = vector_a - vector_b

    # Step 4: Square the differences
    squared_difference = difference ** 2

    # Step 5: Calculate the sum
    sum_squared = np.sum(squared_difference)

    # Step 6: Calculate square root
    distance = np.sqrt(sum_squared)

    return float(distance)

import numpy as np


def normalize_vector(vector):

    # Convert input into NumPy array
    vector = np.asarray(vector, dtype=float)

    # Calculate vector magnitude
    magnitude = np.linalg.norm(vector)

    # Prevent division by zero
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")

    # Divide every element by magnitude
    normalized_vector = vector / magnitude

    return normalized_vector
