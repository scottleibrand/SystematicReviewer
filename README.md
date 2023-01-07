# SystematicReviewer
Tools for doing systematic reviews of scientific literature

## Requirements

### OpenAI API Key

You'll need an OpenAI.com account (sign up at https://beta.openai.com/signup), and to create a key at https://beta.openai.com/account/api-keys

You can provide your key to the script by doing:
```
export OPENAI_API_KEY=sk-YourSuperSecretBigLongStringOfRandomLettersAndNumbers
```
in your terminal prior to running either of the scripts below.

### Input data

These scripts assume that you are starting with a .csv file containing an ArticleURL column containing the URL of each paper's full text HTML or PDF, and optionally Title and Abstract columns.

## Installation

Mac:
```
brew install libmagic
pip install -r requirements.txt
```

## Components

### download_articles_and_embeddings.py

This script downloads papers from a given CSV file, extracts the text from each one, and writes each section of the paper to a separate text file. It then calculates the embedding of each section text file and combines all the section text files and their corresponding embeddings into a single JSON file.

#### Usage

To use this script, run the following command:

```
python download_articles_and_embeddings.py <filename>.csv
```

where `<filename>.csv` is the name of the CSV file containing the papers to be downloaded and extracted.

The CSV should have an ArticleURL column containing the URL of each paper's full text HTML or PDF.

#### Output

The script will output a JSON file containing the text and embeddings of each section of the paper.

### answer_questions.py


This script is used to answer a given question using a combination of OpenAI's GPT-3 model and a set of embeddings.

#### Usage

To use the script, run the following command:

```
python answer_questions.py <filename>.csv combined_text_and_embeddings.json <question_string> [n]
```

Where:

- `csv_file` is the path to the CSV file containing the data to be processed.
- `json_file` is the path to the JSON file containing the embeddings.
- `question_string` is the question to be answered. Be sure to put quotes around it.
- `n` is an optional parameter specifying the number of results to return (defaults to 3).

#### Description

The script takes a CSV file containing data and a JSON file containing embeddings as input. It then uses OpenAI's GPT-3 model to answer the given question.

First, the script processes the CSV file and extracts the URLs from it. For each URL, it extracts the base file name from the URL and searches the JSON file for records whose "file_name" key contains the base file name.

Next, the script uses the embeddings from the JSON file to search for records that are similar to the given question. It then uses the OpenAI GPT-3 model to generate answers for each of the records.

Finally, the script uses the OpenAI GPT-3 model to generate an overall answer based on the intermediate results. The answer is then written to a new CSV file.
