# SystematicReviewer
AI tools for doing systematic reviews of scientific literature

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

It also outputs a .indexed.csv file that adds a SectionsFound column to the original CSV. This indicates the number of sections that were downloaded from the provided URL, for comparison agains min_sections below and to provide an indication of which URLs likely did vs. didn't have full text available at the provided URL.

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
python answer_questions.py <filename>.csv combined_text_and_embeddings.json <questions.csv or string> [top_n_results=1] [min_sections=2]
```

Where:

- `csv_file` is the path to the CSV file containing the data to be processed.
- `json_file` is the path to the JSON file containing the embeddings.
- `questions.csv or string` is either a file containing the questions to be answered, or a single question string in quotes.
- `top_n_results` is an optional parameter (default=1) specifying the number of results to return (defaults to 3).
- `min_sections` is an optional parameter (default=2) specifying the minimum number of sections that must be found to process a paper

The questions.csv must contain a Question column. It can optionally contain a Type column.

The currently supported types are `Categorization` and `Open-ended`. If type is `Categorization`, include as many CategoryN fields as are required to list all of the categories you'd like it to choose between.

Example:
```
Question,Type,Category1,Category2,Category3
What kind of study does this describe?,Categorization,An animal study,A human clinical study,A review or meta-analysis of the literature,
What methods did the study use?,Open-ended,,,,
```

#### Description

The script takes a CSV file containing data and a JSON file containing embeddings as input. It then uses OpenAI's GPT-3 model to answer the given question.

First, the script processes the CSV file and extracts the URLs from it. For each URL, it extracts the base file name from the URL and searches the JSON file for records whose "file_name" key contains the base file name.

Next, the script uses the embeddings from the JSON file to search for records that are similar to the given question. It then uses the OpenAI GPT-3 model to generate answers for each of the records.

Finally, the script uses the OpenAI GPT-3 model to generate an overall answer based on the intermediate results. The answer is then written to a new CSV file.
