# HW1
## Problem 2
Plot time needed for 1000 documents (seconds):
![alt text](figures/tokenizer_time.png)

Average time per document:
- SplitTokenizer: 0.0065 seconds
- RegexTokenize: 0.0077 seconds
- SpaCyTokenizer: 0.6769 seconds 
  
Estimated time to preprocess entire corpus:
- SplitTokenizer: 0.36 hours
- RegexTokenizer: 0.43 hours
- SpaCyTokenizer: 37.6 hours
  
My choice:<br>
SplitTokenizer is the fastest but less accurate, RegexTokenizer balances speed and accuracy, while SpaCyTokenizer is the most accurate but slowest. In practice, I will choose RegexTokenizer, as I need accuracy but I don't want to process the corpus for 37 hours.\

## Problem 5
![alt text](<figures/indexing time and memory usage.png>)
I don't think the positional index may be fit in memory on my own computer, because it consumes more memory than the basic one as the basic one already taken a lot of memory.

## Problem 6
I implemented a Weighted Term Frequency Scorer, which simply adds up the term frequencies for each query term in the document, but multiply each term frequency by a weight. The weight for a term in the query is based on its frequency within the query, meaning more frequently occurring terms in the query are given higher importance.
weight(w_i)=1+log(1+tf(w_i,q))

## Problem 9
It takes me more than two hours to run the evaluation, I think I may not be able to provide the figures.



