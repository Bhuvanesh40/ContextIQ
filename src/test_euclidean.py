
from embeddings.similarity import euclidean_distance

# Create two vectors
vector_a = [1, 2]
vector_b = [4, 6]

# Calculate Euclidean distance
distance = euclidean_distance(vector_a, vector_b)

print("VECTOR A:", vector_a)
print("VECTOR B:", vector_b)

print("\nEUCLIDEAN DISTANCE:", distance)

print("EXPECTED RESULT: 5.0")
