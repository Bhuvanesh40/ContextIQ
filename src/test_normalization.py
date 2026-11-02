
import numpy as np

from embeddings.similarity import normalize_vector


# Original vector
vector = [3, 4]

# Normalize vector
normalized = normalize_vector(vector)

# Calculate original magnitude
original_magnitude = np.linalg.norm(vector)

# Calculate normalized magnitude
normalized_magnitude = np.linalg.norm(normalized)

print("ORIGINAL VECTOR:", vector)

print("ORIGINAL MAGNITUDE:", original_magnitude)

print("\nNORMALIZED VECTOR:", normalized)

print("NORMALIZED MAGNITUDE:", normalized_magnitude)
