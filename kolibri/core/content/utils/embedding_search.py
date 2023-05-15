import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline


class EmbeddingSearch:
    def __init__(self, embeddings, model="sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = embeddings
        self.model = model
        self.pipe = pipeline("feature-extraction", model=self.model, okenizer=self.model)

    def normalized_mean_pooling(self, token_vectors):
        sentences_vectors = [np.mean(tokens, axis=0) for tokens in token_vectors]
        normalized_embeddings = [vector / np.linalg.norm(vector) \
                                 for vector in sentences_vectors]
        return normalized_embeddings

    def encoder(self, texts):
        embeddings = []
        loader = DataLoader(texts, batch_size=32, shuffle=False)
        for inputs in tqdm(loader):
            vectors = self.pipe(inputs)
            vectors = [np.vstack(item) for item in vectors]
            embs = self.normalized_mean_pooling(vectors)
            embeddings.extend(embs)
        return embeddings

    def search(self, query):
        query_embedding = self.encoder([query])
        similarities_bert = cosine_similarity(self.embeddings, query_embedding)

        index_of_highest_scores = np.argsort(similarities_bert, axis=0)[::-1][:5]
        return self.embeddings.iloc[index_of_highest_scores]
