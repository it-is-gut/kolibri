import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline
from kolibri.core.content.models import ContentNodeEmbedding
import ast



class EmbeddingSearch:
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2"):
        self.model = model
        self.pipe = pipeline("feature-extraction", model=self.model, okenizer=self.model)

        # load embeddings from the database
        objects = ContentNodeEmbedding.objects.all().values()
        object = objects[1:]

        # save it to an internal variable
        self.embeddings = [ast.literal_eval(entry["embedding"]) for entry in object]
        self.indices = [entry["id"] for entry in object]

    def normalized_mean_pooling(self, token_vectors):
        sentences_vectors = [np.mean(tokens, axis=0) for tokens in token_vectors]
        normalized_embeddings = [vector / np.linalg.norm(vector) \
                                 for vector in sentences_vectors]
        return normalized_embeddings

    def encoder(self, texts):
        '''
        Use huggingface pipeline class to get vector embeddings for each token,
        then take the mean across tokens to get one vector embbedding per text
        '''

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
        return [self.indices[idx] for idx in list(np.squeeze(index_of_highest_scores))]
