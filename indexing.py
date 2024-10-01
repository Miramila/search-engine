'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import RegexTokenizer, Tokenizer
from collections import Counter, defaultdict
import os
import json
import matplotlib.pyplot as plt
import time
import sys

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(list)  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'

    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata:
            return
        
        update_index = []
        for term in list(self.index.keys()):
            for d,f in self.index[term]:
                if d != docid:
                    update_index.append((d,f))
                else:
                    self.statistics['vocab'][term] -= f
            self.index[term] = update_index
            if not self.index[term]:
                del self.index[term]

        del self.document_metadata[docid]

    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        term_freq = Counter(tokens)
        self.vocabulary.update(term_freq.keys())
        self.document_metadata[docid] = {"length": len(tokens), "unique_tokens": len(term_freq)}

        for term, freq in term_freq.items():
            self.index[term].append((docid, freq))
            self.statistics['vocab'][term] += freq

    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])
    
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(str(doc_id), {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        term_postings = self.index.get(term, [])
        term_count = self.statistics['vocab'].get(term, 0)
        doc_frequency = len(term_postings)

        return {
            "term_count": term_count,
            "doc_frequency": doc_frequency
        }

    def get_statistics(self) -> dict[str, int]:
        number_of_documents = len(self.document_metadata)
        total_token_count = sum(meta['length'] for meta in self.document_metadata.values())
        unique_token_count = len(self.vocabulary)
        stored_total_token_count = sum(self.statistics['vocab'].values())
        mean_document_length = total_token_count / number_of_documents if number_of_documents > 0 else 0

        return {
            "unique_token_count": unique_token_count,
            "total_token_count": total_token_count,
            "stored_total_token_count": stored_total_token_count,
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_document_length
        }

    def save(self, index_directory_name: str) -> None:
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)

        with open(os.path.join(index_directory_name, 'index.json'), 'w') as f:
            json.dump(self.index, f)

        with open(os.path.join(index_directory_name, 'metadata.json'), 'w') as f:
            json.dump(self.document_metadata, f)

    def load(self, index_directory_name: str) -> None:
        with open(os.path.join(index_directory_name, 'index.json'), 'r') as f:
            self.index = json.load(f)

        with open(os.path.join(index_directory_name, 'metadata.json'), 'r') as f:
            self.document_metadata = json.load(f)
        
        self.vocabulary = set(self.index.keys())
        for term, postings in self.index.items():
            self.statistics['vocab'][term] = sum(freq for _, freq in postings)


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()

    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata:
            return
        
        update_index = []
        for term in list(self.index.keys()):
            for d,f,p in self.index[term]:
                if d != docid:
                    update_index.append((d,f,p))
                else:
                    self.statistics['vocab'][term] -= f
            self.index[term] = update_index
            if not self.index[term]:
                del self.index[term]

        del self.document_metadata[docid]
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        term_positions = defaultdict(list)

        for position, term in enumerate(tokens):
            if term is not None:
                term_positions[term].append(position)

        self.vocabulary.update(term_positions.keys())
        self.document_metadata[docid] = {"length": len(tokens), "unique_tokens": len(term_positions)}

        for term, positions in term_positions.items():
            self.index[term].append((docid, len(positions), positions))
            self.statistics['vocab'][term] += len(positions)

        
    

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''


    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                        document_preprocessor: Tokenizer, stopwords: set[str],
                        minimum_word_frequency: int, text_key = "text",
                        max_docs: int = -1, ) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        
        '''
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            index = SampleIndex()

        if max_docs == 0:
            return index

        global_term_freq = Counter()
        documents = []
        with open(dataset_path, 'r') as f:
            for line in f:
                documents.append(json.loads(line))

        for _, document in enumerate(documents):
            doc_id = document["docid"]
            if 0 <= max_docs <= doc_id:
                break

            text = document.get(text_key, "")
            tokens = document_preprocessor.tokenize(text)
            index.add_doc(doc_id, tokens)
            global_term_freq.update(tokens)



        # stopwords filter
        if stopwords:
            for term in list(index.index.keys()):
                if term in stopwords:
                    if term in index.index:
                        postings = index.get_postings(term)
                        for post in postings:
                            if post[0] in index.document_metadata:
                                index.document_metadata[doc_id]['unique_tokens'] -= 1
                        del index.index[term]
                        index.vocabulary.discard(term)
                        del index.statistics['vocab'][term]

        # minimum word frequency filter
        if minimum_word_frequency > 0:
            terms_to_remove = {term for term, freq in global_term_freq.items() if freq < minimum_word_frequency}

            for term in terms_to_remove:
                if term in index.index:
                    postings = index.get_postings(term)
                    for post in postings:
                        if post[0] in index.document_metadata:
                            index.document_metadata[doc_id]['unique_tokens'] -= 1

                    del index.index[term]
                    index.vocabulary.discard(term)
                    del index.statistics['vocab'][term]

        return index
        


# TODO for each inverted index implementation, use the Indexer to create an index with the first 10, 100, 1000, and 10000 documents in the collection (what was just preprocessed). At each size, record (1) how
# long it took to index that many documents and (2) using the get memory footprint function provided, how much memory the index consumes. Record these sizes and timestamps. Make
# a plot for each, showing the number of documents on the x-axis and either time or memory
# on the y-axis.

'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1
    
    def save(self):
        print('Index saved!')

if __name__ == '__main__':
    documents = []
    with open('data/wikipedia_200k_dataset.jsonl', 'r') as f:
        for _ in range(10000):
            line = f.readline()
            documents.append(json.loads(line))
            
    tokenizer = RegexTokenizer('\w+')
    tokenized_documents = [tokenizer(doc["text"]) for doc in documents]

    def index_documents(index_class, document_samples):
        index = index_class()
        start_time = time.time()
        for docid, tokens in enumerate(document_samples):
            index.add_doc(docid, tokens)
        end_time = time.time()
        time_taken = end_time - start_time
        memory_usage = sys.getsizeof(index)
        return time_taken, memory_usage
    
    # Prepare data for plotting
    doc_counts = [10, 100, 1000, 10000]
    basic_times = []
    basic_memories = []
    positional_times = []
    positional_memories = []
    
    # Index documents using both Basic and Positional Inverted Indexes
    for count in doc_counts:
        sample_documents = tokenized_documents[:count]
        
        # Basic Inverted Index
        time_taken, memory_usage = index_documents(BasicInvertedIndex, sample_documents)
        basic_times.append(time_taken)
        basic_memories.append(memory_usage)
    
        # Positional Inverted Index
        time_taken, memory_usage = index_documents(PositionalInvertedIndex, sample_documents)
        positional_times.append(time_taken)
        positional_memories.append(memory_usage)
    
    # Plotting results
    plt.figure(figsize=(12, 5))
    
    # Plot indexing time
    plt.subplot(1, 2, 1)
    plt.plot(doc_counts, basic_times, label='Basic Inverted Index', marker='o')
    plt.plot(doc_counts, positional_times, label='Positional Inverted Index', marker='o')
    plt.xlabel('Number of Documents')
    plt.ylabel('Indexing Time (seconds)')
    plt.title('Indexing Time vs. Number of Documents')
    plt.legend()
    
    # Plot memory usage
    plt.subplot(1, 2, 2)
    plt.plot(doc_counts, basic_memories, label='Basic Inverted Index', marker='o')
    plt.plot(doc_counts, positional_memories, label='Positional Inverted Index', marker='o')
    plt.xlabel('Number of Documents')
    plt.ylabel('Memory Usage (bytes)')
    plt.title('Memory Usage vs. Number of Documents')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


