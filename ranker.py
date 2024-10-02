"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
from math import sqrt, log
from collections import Counter


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query

        # 2. Fetch a list of possible documents from the index

        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]

        tokens = self.tokenize(query)
        query_word_counts = Counter(tokens)

        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Get relevant documents
        candidate_docs = set()
        for term in query_word_counts:
            postings = self.index.index.get(term, [])
            candidate_docs.update(doc_id for doc_id, _ in postings)


        # Score documents
        scores = []
        for docid in candidate_docs:
            doc_word_counts = {}
            for term, doc_freq in self.index.index.items():
                for doc_id, count in doc_freq:
                    if doc_id == docid:
                        if term not in doc_word_counts:
                            doc_word_counts[term] = 0
                        doc_word_counts[term] += count
                        break
            score = self.scorer.score(docid, doc_word_counts, query_word_counts)
            scores.append((docid, score))

        # Sort and return results
        return sorted(scores, key=lambda x: x[1], reverse=True)


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        dot_product = sum(doc_word_counts.get(word, 0) * query_word_counts.get(word, 0) for word in query_word_counts)
        # doc_magnitude = sqrt(sum(doc_word_counts.get(word, 0) ** 2 for word in query_word_counts))
        # print("doc_magnitude",doc_magnitude)
        # query_magnitude = sqrt(sum(count ** 2 for count in query_word_counts.values()))
        # print("query_magnitude", query_magnitude)

        # if doc_magnitude == 0 or query_magnitude == 0:
        #     return 0.0``
        
        return dot_product

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        doc_len = self.index.document_metadata[docid]["length"]
        mu = self.parameters['mu']
        score = 0.0
        for q_term in query_word_counts:
            if q_term and q_term in doc_word_counts:
                postings = self.index.get_postings(q_term)
                doc_tf = doc_word_counts[q_term]

                if doc_tf > 0:
                    query_tf = query_word_counts[q_term]
                    p_wc = sum([doc[1] for doc in postings]) / self.index.get_statistics()['total_token_count']
                    tfidf = log(1 + (doc_tf / (p_wc * mu)))
                    score += query_tf * tfidf
        score = score + len(query_word_counts) * log(mu / (doc_len+mu))
        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score    

        # 4. Return score
        score = 0.0
        total_docs = len(self.index.document_metadata)
        avgdl = sum(metadata["length"] for metadata in self.index.document_metadata.values()) / total_docs
        doc_len = self.index.document_metadata[docid]["length"]

        for term in query_word_counts:
            if term in doc_word_counts:
                df = len(self.index.index.get(term, []))
                idf = log((total_docs - df + 0.5) / (df + 0.5))
                tf = doc_word_counts[term]
                score += idf * ((self.k1 + 1) * tf) / (self.k1 * ((1 - self.b) + self.b * (doc_len / avgdl)) + tf)
        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        score = 0.0
        total_docs = len(self.index.document_metadata)
        avgdl = sum(metadata["length"] for metadata in self.index.document_metadata.values()) / total_docs
        doc_len = self.index.document_metadata[docid]["length"]

        for term in query_word_counts:
            if term in doc_word_counts:
                tf = 1 + log(1 + log(doc_word_counts[term]))
                df = len(self.index.index.get(term, []))
                idf = log((total_docs + 1) / df)
                normalization = 1 / (1 - self.b + self.b * (doc_len / avgdl))
                score += tf * idf * normalization
        return score


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        score = 0.0
        total_docs = len(self.index.document_metadata)

        for term in query_word_counts:
            if term in doc_word_counts:
                tf = log(doc_word_counts[term] + 1)
                df = len(self.index.index.get(term, []))
                idf = log(total_docs / df) + 1
                score += tf * idf
        return score


# TODO Implement your own ranker with proper heuristics
class WeightedTermFrequencyScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0.0
        for term in query_word_counts:
            if term in doc_word_counts:
                weight = 1 + log(1 + query_word_counts[term])
                score += weight * doc_word_counts[term]
        return score
