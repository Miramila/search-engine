from math import log2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json

from document_preprocessor import RegexTokenizer
from indexing import IndexType, Indexer
from ranker import BM25, PivotedNormalization, TF_IDF, DirichletLM, WordCountCosineSimilarity, WeightedTermFrequencyScorer

"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    relevant_count = 0
    precision_at_k_sum = 0

    for i in range(len(search_result_relevances)):
        if search_result_relevances[i] == 1:
            relevant_count += 1
            if i < cut_off:
                precision_at_k_sum += relevant_count / (i + 1)

    return precision_at_k_sum / relevant_count if relevant_count > 0 else 0


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    def dcg(relevances: list[float], cut_off: int) -> float:
        return sum((rel / log2(i + 1)) if i !=0 else rel for i, rel in enumerate(relevances[:cut_off]))

    actual_dcg = dcg(search_result_relevances, cut_off)
    ideal_dcg = dcg(ideal_relevance_score_ordering, cut_off)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    # Load relevance dataset
    relevance_data = pd.read_csv(relevance_data_filename, sep=',')
    queries = relevance_data['query'].unique()
    map_scores = []
    ndcg_scores = []

    for query in queries:
        query_data = relevance_data[relevance_data['query'] == query]
        relevance_dict = dict(zip(query_data['docid'], query_data['rel']))

        results = ranker.query(query)
        ranked_docs = [docid for docid, _ in results]

        relevance_scores = [relevance_dict.get(docid, 0) for docid in ranked_docs]
        binary_relevance_scores = [1 if rel > 0 else 0 for rel in relevance_scores]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)

        map_scores.append(map_score(binary_relevance_scores, cut_off=10))
        ndcg_scores.append(ndcg_score(relevance_scores, ideal_relevance_scores, cut_off=10))

    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return {
        'map': avg_map,
        'ndcg': avg_ndcg,
        'map_list': map_scores,
        'ndcg_list': ndcg_scores
    }

def process_raw_text(dataset_name):
    raw_text = {}
    with open(dataset_name) as f:
        for line in f:
            data = json.loads(line)
            raw_text[data['docid']] = data['text']
    return raw_text



if __name__ == '__main__':
    # set up
    index = Indexer.create_index(IndexType.BasicInvertedIndex, './data/wikipedia_200k_dataset.jsonl', RegexTokenizer('\\w+'), set(), 0)

    raw_text = process_raw_text('./data/wikipedia_200k_dataset.jsonl')

    rankers = [WordCountCosineSimilarity(index, raw_text), TF_IDF(index, raw_text), BM25(index, raw_text), PivotedNormalization(index, raw_text), DirichletLM(index, raw_text), WeightedTermFrequencyScorer(index, raw_text)]

    
    ranker_results = {}
    for ranker in rankers:
        ranker_results[ranker.__class__.__name__] = run_relevance_tests('./data/relevance.test.csv', ranker)
    
    ranker_names = list(ranker_results.keys())
    
    # Prepare data for table
    avg_map_scores = [ranker_results[ranker]['map'] for ranker in ranker_names]
    avg_ndcg_scores = [ranker_results[ranker]['ndcg'] for ranker in ranker_names]
    table_data = pd.DataFrame([avg_map_scores, avg_ndcg_scores], index=['MAP', 'NDCG'], columns=ranker_names)
    
    print("Table: Average Scores")
    print(table_data)
    
    # Prepare data for plotting
    map_scores_list = [(ranker, score) for ranker in ranker_names for score in ranker_results[ranker]['map_list']]
    ndcg_scores_list = [(ranker, score) for ranker in ranker_names for score in ranker_results[ranker]['ndcg_list']]
    
    map_df = pd.DataFrame(map_scores_list, columns=['Ranker', 'Score'])
    ndcg_df = pd.DataFrame(ndcg_scores_list, columns=['Ranker', 'Score'])
    
    # Plot MAP scores
    plt.figure(figsize=(12, 5))
    sns.violinplot(x='Ranker', y='Score', data=map_df)
    plt.title('MAP Scores for Different Ranking Functions')
    plt.ylabel('MAP Score')
    plt.xticks(rotation=45)
    plt.show()
    
    # Plot NDCG scores
    plt.figure(figsize=(12, 5))
    sns.violinplot(x='Ranker', y='Score', data=ndcg_df)
    plt.title('NDCG Scores for Different Ranking Functions')
    plt.ylabel('NDCG Score')
    plt.xticks(rotation=45)
    plt.show()
