

## Code Structure
### Files 

- app.py: The main file that contains the FastAPI code. 
- models.py: The file that contains some data classes that are used in the project. 
- document_preprocessor.py: The file that contains the code to preprocess the documents. 
- indexing.py: The file that contains the code to index the documents. 
- ranker.py: The file that contains the code to rank the documents. 
- relevance.py: The file that contains the code to compute metrics like MAP and NDCG. 
- pipeline.py: The file that contains the code to run the search engine pipeline.

### Data
- data  
    - multi_word_expressions.txt: The file that contains the multi-word expressions. 
    - wikipedia_200k_dataset.jsonl.gz: The file that contains the Wikipedia dataset. 
    - stopwords.txt: The file that contains the stopwords. 
    - relevance.test.csv: The file that contains the test data for the relevance scorer. 

The `wikipedia_200k_dataset.jsonl.gz` dataset contains 200 thousand Wikipedia articles. It is in JSONL format where each lines is a separate JSON of the following format. 

```
{
    "docid": <document id>
    "title": <document title>
    "text": <the entire text of the document>
    "categories": [<each Wikipedia category>]
}
```

## How to run the Search Engine

This is a Python 3.11 FastAPI project with the necessary requirements added to the `requirements.txt`.

### Environment Setup

After downloading the `HW1.zip` file from canvas, you will unzip the file, which gives you a `HW1` folder. You will also download `multi_word_expressions.txt` from canvas. Put it in the `HW1` folder. 

Now, we will create a virtual environment for your search engine project.

1. Install Python 3.11

    Follow the link [Python 3.11](https://www.python.org/downloads/release/python-3112/) to install Python 3.11. After installation, you can check the version of your Python by running the following command.

    For macOS and Linux, you will be running this in Terminal.

    For Windows, you will be running this in Command Prompt.

    ```
    python --version
    ```

2. Create a virtual environment

    Get the path of your python installation.

    Use the following command to create a virtual environment specifically for SI 650.

    ```
    <path of your python installation>/python -m venv si650
    ```

    If you installed Python 3.11 without overwriting the default Python version, use the following command:

    ```
    python3.11 -m venv si650
    ```

    This will create a folder `si650` inside the folder you navigated to. In the example above, this will create a virtual environment (a folder) inside `my_project`.

3. Activate the environment 

    - For Windows, run
    ```
    si650/bin/activate
    ```

    - For Mac and others, run
    ```
    source si650/bin/activate
    ```

4. Install the requirements

    After activating the virtual environment, navigate to the `HW1` folder by using the cd command. 

    Run the following command:

    ```
    python -m pip install -r requirements.txt
    ```
    This will install the libraries you need to start the server.

### Start the Server

After you have all of these files and the necessary Python requirements installed in your environment, run 

```
python3 -m uvicorn app:app
```

to start the server.

It will give you an address that starts with `http://`. Copy and paste it in your browser. 

NOTE: To get a functioning search engine, you need to implement at least one of the preprocessor, indexer, and ranker. Otherwise you would get an empty page with no search results. 


### Deactivate the virtual environment

After you are done running the server, you can use this command to deactivate the virtual environment:

```
deactivate
```

After running the deactivate command, your virtual environment will be deactivated, and you'll return to the global Python environment. You'll see that the virtual environment's name (in our case, `(si650)`) disappears from your command prompt, indicating that you are no longer in the virtual environment.


More comments are present in the code files themselves. And if you have trouble understanding parts of the code, please ping any of the GSIs for the course on Slack or create a post on Piazza. 
