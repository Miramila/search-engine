from document_preprocessor import RegexTokenizer
from indexing import IndexType, Indexer
from ranker import BM25, PivotedNormalization, TF_IDF, DirichletLM, WordCountCosineSimilarity, WeightedTermFrequencyScorer


index = Indexer.create_index(IndexType.BasicInvertedIndex, './data/wikipedia_200k_dataset.jsonl', RegexTokenizer('\\w+'), set(), 0)
index.save('data_processed')