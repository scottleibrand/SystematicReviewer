# Description: This script takes a CSV file containing a list of URLs,
# a JSON file containing the embeddings, a question string, and a number n,
# and returns the top n answers to the question from the embeddings.

import os
import re
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import json
import sys
import numpy as np
import tiktoken
import openai

def search_embeddings(df, question_string, n):
    question_embedding = get_embedding(
        question_string,
        engine="text-embedding-ada-002"
    )
    # Add a similarity column to the dataframe calculated using the cosine similarity function
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, question_embedding))
    
    res = (
        df.sort_values("similarities", ascending=False)
        .head(n)
        .text
        # Get all the elements of the list
        .apply(lambda x: x)
    )
    return res

def convert_to_json(res):
  # Convert the dataframe to a list of dictionaries, where each dictionary
  # represents a message with 'text' as the key and the message text as the value
  ticket = [{'text': text} for text in res]

  # Create a JSON object with the list of dictionaries as the value for the 'messages' key
  result = {'ticket': ticket}

  # Return the JSON object
  return result



def get_nth_result(results, n):
  # Get the list of messages from the results
  ticket = results['ticket']

  # Get the Nth message from the list
  nth_message = ticket[n]

  # Return the text of the Nth message
  return nth_message['text']

def ask_gpt(prompt, model_engine="text-davinci-003", max_tokens=3000):
    # Get the API key from the environment variable
    api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = api_key

    # Set the model to use, if not specified
    if model_engine is None:
        model_engine = "text-davinci-003"

    # Set the temperature for sampling
    temperature = 0

    # Set the max token count for the answer
    max_tokens = 100
    #if model_engine == "text-davinci-003":
    #    max_tokens = 1000
    #else:
    #    max_tokens = 500

    # Generate completions
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Get the summary from the first completion
    answer = completions.choices[0].text

    return answer


def main(csv_file, json_file, question_string, n):
    # Process the csv file
    df = pd.read_csv(csv_file)
    urls = df['ArticleURL']
    answers = []
    for url in urls:
        # Skip "nan" URLs
        if pd.isna(url):
            answers.append('')
            continue
        print(f"Processing {url}...")
        base_file_name = os.path.basename(url)
        if base_file_name == '':
            base_file_name = url.rsplit('/', 2)[-2]
        # If it's still longer than 100 characters, truncate it
        if len(base_file_name) > 100:
            base_file_name = base_file_name[:100]

        # Remove non-alphanumeric, non-period and non-underscore characters from the file name
        base_file_name = re.sub(r'[^\w_.]', '', base_file_name)
        
        # Load the json file
        with open(json_file) as f:
            data = json.load(f)

        # Store the intermediate results in a list
        intermediate_results = []

        # Use this row's value from the Title and Abstract column of the csv, if present, as the context
        if 'Title' in df.columns:
            # Look up the row with the same URL as the current URL
            title_df = df[df['ArticleURL'] == url]
            # Extract the value from the Title column as a string
            if len(title_df) != 0:
                context = title_df['Title'].values[0]

        if 'Abstract' in df.columns:
            # Look up the row with the same URL as the current URL
            abstract_df = df[df['ArticleURL'] == url]
            # Extract the value from the Abstract column as a string
            if len(abstract_df) != 0:                    
                context = context + "\n" + abstract_df['Abstract'].values[0]

                #context = df['Abstract']
                #print(context)
                prompt = "Given the following context:\n" + context + \
                    "\nPlease answer the following question if it is answered by the context.\n" + \
                    "Question: " + question_string + "\n" + \
                    "If the context is not relevant to the question, reply with 'The context is not relevant to the question.'"
                #print(prompt)
                answer = ask_gpt(prompt)
                #print(answer)
                intermediate_results.append(answer)

        
        # Find all records whose "file_name" key contains the base_file_name extracted from the URL
        matching_records = []
        print(f"Searching for records with file names containing {base_file_name}...")
        for record in data:
            if base_file_name in record['file_name']:
                #print(f"Found record with file name {record['file_name']}.")
                matching_records.append(record)
            
        print(f"Found {len(matching_records)} matching records.")
        if len(matching_records) == 0:
            print("No matching records found. Skipping...")
            answers.append('')
            continue
        # Load all of those matching records into a pandas dataframe
        matching_df = pd.DataFrame(matching_records)

        # Search the embeddings
        matching_res = search_embeddings(matching_df, question_string, n)



        # Loop through each result
        for i in range(len(matching_res)):
            # Convert the result to a JSON object
            results = convert_to_json(matching_res)
            result = get_nth_result(results, i)
            results_string = json.dumps(result)
            #print(results_string)

            # Get the token length of the string
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(results_string)
            token_count = len(tokens)
            # print the length of the string in characters and tokens
            #print("String length: " + str(len(results_string)) + " characters, "Token count: " + str(token_count))
            print(f"String length: {len(results_string)} characters, Token count: {token_count}")
            #print(results_string)

    
            prompt = "Given the following context:\n" + results_string + \
                "\nPlease answer the following question if it is answered by the context.\n" + \
                "Question: " + question_string + "\n" + \
                "If the context is not relevant to the question, reply with 'The context is not relevant to the question.'"
            #print(prompt)
            answer = ask_gpt(prompt)
            #print(answer)
            intermediate_results.append(answer)

        # Ask GPT to provide an overall answer based on the intermediate results
        prompt = "Given the following " + str(len(intermediate_results)) + " answers:\n" + \
            "\n".join(intermediate_results) + \
            "\nPlease provide an overall answer to the question:\n" + \
            question_string
        print(prompt)
        answer = ask_gpt(prompt)
        # Remove empty lines from the answer
        answer = re.sub(r'\n\s*\n', '', answer)
        print(answer)
        answers.append(answer)
    # Write out the results to a new csv
    df[question_string] = answers
    df.to_csv(csv_file.replace('.csv', '.out.csv'), index=False)
    
    
        #return res


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <csv_file> <json_file> <question_string> [n]")
        sys.exit(1)
    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    question_string = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    main(csv_file, json_file, question_string, n)
    