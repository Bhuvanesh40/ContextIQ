
from embeddings.similarity import dot_product

# Create two sample vectors
vector_a = [2, 3]
vector_b = [4, 5]

# Calculate dot product
result = dot_product(vector_a, vector_b)

print("VECTOR A:", vector_a)
print("VECTOR B:", vector_b)

print("\nDOT PRODUCT:", result)

# Expected result
print("EXPECTED RESULT: 23")
