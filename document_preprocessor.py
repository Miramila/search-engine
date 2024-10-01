"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""
from nltk.tokenize import RegexpTokenizer
import spacy
import time
import matplotlib.pyplot as plt
import json


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        output_tokens = []
        i = 0

        if not self.multiword_expressions:
            output_tokens = input_tokens
        else:
            # check if any multi-word expression matches
            while i < len(input_tokens):
                matched = False
                for mwe in sorted(self.multiword_expressions, key=len, reverse=True): # ensure the longest first
                    mwe_tokens = mwe.split()
                    if (
                        i + len(mwe_tokens) <= len(input_tokens) and
                        [token for token in input_tokens[i:i+len(mwe_tokens)]] == mwe_tokens
                    ):
                        matched_tokens = input_tokens[i:i+len(mwe_tokens)]
                        output_tokens.append(" ".join(matched_tokens))
                        i += len(mwe_tokens)
                        matched = True
                        break
    
                if not matched:
                    output_tokens.append(input_tokens[i])
                    i += 1

        if self.lowercase:
            output_tokens = [token.lower() for token in output_tokens]

        return output_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.token_regex = token_regex
        self.tokenizer = RegexpTokenizer(self.token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        tokens = self.tokenizer.tokenize(text)
        return self.postprocess(tokens)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single tokens while keeping the order.
    
        Args:
            text: An input text you want to tokenize
    
        Returns:
            A list of tokens
        """
        doc = self.nlp(text)
        tokens = []
        
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.ent_iob_ == "B":  # Beginning of a named entity
                entity_tokens = [token.text]
                i += 1
                while i < len(doc) and doc[i].ent_iob_ == "I" and doc[i].ent_type_ == token.ent_type_:
                    entity_tokens.append(doc[i].text)
                    i += 1
                tokens.append(" ".join(entity_tokens))
            else:
                tokens.append(token.text)
                i += 1

        return self.postprocess(tokens)
    
# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    # read in the multi-word expressions
    mwe_filepath = 'tests/multi_word_expressions.txt'
    mwe_list = []
    with open(mwe_filepath, 'r') as f: 
        lines = f.readlines()
        for line in lines:
            mwe_list.append(line.strip())

    # read in 1000 line of sample documents
    jsonl_file_path = 'data/wikipedia_200k_dataset.jsonl'  # Replace with your JSONL file path
    documents = []

    with open(jsonl_file_path, 'r') as f:
        for _ in range(1000):
            line = f.readline()
            if not line:
                break
            json_line = json.loads(line)
            documents.append(json_line['text'])

    # Initialize tokenizers
    split_tokenizer = SplitTokenizer(lowercase=True, multiword_expressions=["artificial intelligence", "machine learning"])
    regex_tokenizer = RegexTokenizer(lowercase=True, multiword_expressions=["artificial intelligence", "machine learning"])
    spacy_tokenizer = SpaCyTokenizer(lowercase=True, multiword_expressions=["artificial intelligence", "machine learning"])
    
    # Function to measure time taken for tokenization
    def measure_time(tokenizer, documents):
        start_time = time.time()
        for doc in documents:
            tokenizer.tokenize(doc)
        end_time = time.time()
        return end_time - start_time
    
    # Measure the time taken for each tokenizer
    split_time = measure_time(split_tokenizer, documents)
    regex_time = measure_time(regex_tokenizer, documents)
    spacy_time = measure_time(spacy_tokenizer, documents)
    
    # Plot the time taken for each tokenizer
    tokenizers = ['SplitTokenizer', 'RegexTokenizer', 'SpaCyTokenizer']
    times = [split_time, regex_time, spacy_time]
    
    plt.figure(figsize=(10, 6))
    plt.bar(tokenizers, times, color=['blue', 'green', 'red'])
    plt.xlabel('Tokenizer Type')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Time Taken to Tokenize 1000 Documents')
    plt.show()
    
    # Average time per document
    average_times = [t / 1000 for t in times]
    print(f"Average time per document (SplitTokenizer): {average_times[0]:.4f} seconds")
    print(f"Average time per document (RegexTokenizer): {average_times[1]:.4f} seconds")
    print(f"Average time per document (SpaCyTokenizer): {average_times[2]:.4f} seconds")
    
    # Estimate time to preprocess an entire corpus of 1,000,000 documents
    num_documents = 200000
    estimated_times = [avg_time * num_documents for avg_time in average_times]
    print(f"Estimated time to preprocess 1,000,000 documents:")
    print(f" - SplitTokenizer: {estimated_times[0] / 3600:.2f} hours")
    print(f" - RegexTokenizer: {estimated_times[1] / 3600:.2f} hours")
    print(f" - SpaCyTokenizer: {estimated_times[2] / 3600:.2f} hours")
