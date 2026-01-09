from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")

model.save("./models/Arabic-Triplet")