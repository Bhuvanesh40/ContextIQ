
from embeddings.embedding_model import generate_embeddings
from embeddings.similarity import cosine_similarity


sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]

# Generate embeddings
vectors = generate_embeddings(sentences)

# Compare sentence 1 and sentence 2
similarity_1 = cosine_similarity(
    vectors[0],
    vectors[1]
)

# Compare sentence 1 and sentence 3
similarity_2 = cosine_similarity(
    vectors[0],
    vectors[2]
)

print("\nSIMILARITY RESULTS")

print(
    "Sentence 1 vs Sentence 2:",
    similarity_1
)

print(
    "Sentence 1 vs Sentence 3:",
    similarity_2
)
