
from embeddings.embedding_model import generate_embeddings
from embeddings.similarity import euclidean_distance

sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]

# Generate embeddings
vectors = generate_embeddings(sentences)

# Calculate distances
distance_1 = euclidean_distance(
    vectors[0],
    vectors[1]
)

distance_2 = euclidean_distance(
    vectors[0],
    vectors[2]
)

print("\nEUCLIDEAN DISTANCE RESULTS")

print("AI-related sentences:", distance_1)

print("AI vs Pizza:", distance_2)
