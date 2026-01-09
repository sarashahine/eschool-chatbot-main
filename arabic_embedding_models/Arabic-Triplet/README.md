---
base_model: aubmindlab/bert-base-arabertv02
datasets:
- akhooli/arabic-triplets-1m-curated-sims-len
language:
- ar
library_name: sentence-transformers
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- sentence-transformers
- transformers.js
- transformers
- sentence-similarity
- feature-extraction
- dataset_size:75000
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
- mteb
---

# Arabic Triplet Matryoshka V2 Model [ATM2]

![image/png](https://cdn-uploads.huggingface.co/production/uploads/628f7a71dd993507cfcbe587/FrLQzFUJ3grEUOdONWGME.png)

## Model Description

Arabic-Triplet-Matryoshka-V2-Model is a state-of-the-art Arabic language embedding model based on the [sentence-transformers](https://www.SBERT.net) framework. It is fine-tuned from [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02) and specifically designed to capture the rich semantic nuances of Arabic text.

It is described in detail in the paper [GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity with Hybrid Loss Training](https://huggingface.co/papers/2505.24581).

This model maps sentences and paragraphs to a 768-dimensional dense vector space, enabling high-quality semantic text operations including:
- Semantic textual similarity
- Semantic search
- Paraphrase mining
- Text classification
- Clustering
- Information retrieval
- Question answering

## Key Features

- **State-of-the-Art Performance**: Achieved 0.85 on STS17 and 0.64 on STS22.v2 with an average score of 74.5, making it the leading Arabic embedding model currently available.
- **MatryoshkaLoss Training**: Utilizes nested embedding learning techniques to create hierarchical embeddings at multiple resolutions.
- **Optimization**: Trained for 3 epochs with a final training loss of 0.718.
- **Full Arabic Language Support**: Designed specifically to handle the complexity and morphological richness of Arabic language.

## Training Details

The model was trained using a combination of two loss functions:
- **MatryoshkaLoss**: Enables the creation of nested embeddings at multiple resolutions, allowing for efficient and adaptable representations.
- **MultipleNegativesRankingLoss**: Enhances the model's ability to discriminate between semantically similar and dissimilar text pairs.

Training parameters:
- **Base model**: aubmindlab/bert-base-arabertv02
- **Dataset**: akhooli/arabic-triplets-1m-curated-sims-len (1M samples)
- **Epochs**: 3
- **Final Loss**: 0.718
- **Embedding Dimension**: 768

## Performance

The model demonstrates exceptional performance on standard Arabic semantic textual similarity benchmarks:

| Model                                    | Dim  | # Params. | STS17 | STS22-v2 | Average |
|------------------------------------------|------|-----------|-------|----------|---------|
| **Arabic-Triplet-Matryoshka-V2**             | 768  | 135M      | 85    | 64       | 75      |
| Arabert-all-nli-triplet-Matryoshka       | 768  | 135M      | 83    | 64       | 74      |
| AraGemma-Embedding-300m                  | 768  | 303M      | 84    | 62       | 73      |
| GATE-AraBert-V1                          | 767  | 135M      | 83    | 63       | 73      |
| Marbert-all-nli-triplet-Matryoshka       | 768  | 163M      | 82    | 61       | 72      |
| Arabic-labse-Matryoshka                  | 768  | 471M      | 82    | 61       | 72      |
| AraEuroBert-Small                        | 768  | 210M      | 80    | 61       | 71      |
| E5-all-nli-triplet-Matryoshka            | 384  | 278M      | 80    | 60       | 70      |
| text-embedding-3-large                   | 3072 | -         | 81    | 59       | 70      |
| Arabic-all-nli-triplet-Matryoshka        | 768  | 135M      | 82    | 54       | 68      |
| AraEuroBert-Mid                          | 1151 | 610M      | 83    | 53       | 68      |
| paraphrase-multilingual-mpnet-base-v2    | 768  | 135M      | 79    | 55       | 67      |
| AraEuroBert-Large                        | 2304 | 2.1B      | 79    | 55       | 67      |
| text-embedding-ada-002                   | 1536 | -         | 71    | 62       | 66      |
| text-embedding-3-small                   | 1536 | -         | 72    | 57       | 65      |


This represents the current state-of-the-art for Arabic embedding models, outperforming previous approaches by a significant margin.

## Use Cases

This model is particularly well-suited for:
- **Information Retrieval**: Enhancing search capabilities for Arabic content.
- **Document Similarity**: Identifying similar documents or text passages.
- **Text Classification**: Powering classification systems for Arabic content.
- **Question Answering**: Supporting Arabic QA systems with improved semantic understanding.
- **Semantic Clustering**: Organizing Arabic text data based on meaning.
- **Cross-lingual Applications**: When combined with other language models for multilingual applications.

## Usage Examples

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")
# Run inference
sentences = [
    'SENTENCE 1',
    'SENTENCE 2',
    'SENTENCE 3',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

## Limitations

Despite its strong performance, users should be aware of the following limitations:
- The model may not perform optimally on highly technical or domain-specific Arabic text that was underrepresented in the training data.
- As with all embedding models, performance may vary across different Arabic dialects and regional variations.
- The model is optimized for semantic similarity tasks and may require fine-tuning for other specific applications.

## Ethical Considerations

This model is intended for research and applications that benefit Arabic language processing. Users should be mindful of potential biases that may exist in the training data and the resulting embeddings. We encourage responsible use of this technology and welcome feedback on ways to improve fairness and representation.

## Citation

If you use the Arabic Matryoshka Embeddings Model in your research or applications, please cite it as follows:

```bibtex
@article{nacar2025gate,
  title={GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity with Matryoshka Representation Learning and Hybrid Loss Training},
  author={Nacar, Omer and Koubaa, Anis and Sibaee, Serry and Al-Habashi, Yasser and Ammar, Adel and Boulila, Wadii},
  journal={arXiv preprint arXiv:2505.24581},
  year={2025}
}
```

## Acknowledgements

We would like to acknowledge [AraBERT](https://github.com/aub-mind/arabert) for the base model and [akhooli](https://huggingface.co/akhooli) for the valuable dataset that made this work possible.