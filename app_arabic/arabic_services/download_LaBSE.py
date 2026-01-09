from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/LaBSE")

model.save("./models/LaBSE")