# SystematicReviewer
Tools for doing systematic reviews of scientific literature

## Installation

Mac:
```
brew install libmagic
pip install -r requirements.txt
```

## Components

### download_articles_and_embeddings.py

This script downloads articles from a given CSV file, extracts the text from each article, and writes each section of the article to a separate text file. It then calculates the embedding of each section text file and combines all the section text files and their corresponding embeddings into a single JSON file.

#### Usage

To use this script, run the following command:

```
python download_articles_and_embeddings.py <filename>.csv
```

where `<filename>.csv` is the name of the CSV file containing the articles to be downloaded and extracted.

The CSV should have an ArticleURL column containing the URL of each article's full text HTML or PDF.

#### Output

The script will output a JSON file containing the text and embeddings of each section of the article.
