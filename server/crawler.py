import time
import requests
from bs4 import BeautifulSoup
import re
from transformers import DistilBertTokenizer, DistilBertModel  # I used DistiLBERT rather than BERT because the latter was too computationally expensive
from urllib.parse import urljoin, urlparse, unquote
import numpy as np  # These two libraries are used to calculate the cosine similarities between two vectors
from numpy.linalg import norm
import torch # PyTorch is a library used for machine learning

tokenizer = DistilBertTokenizer.from_pretrained( "distilbert-base-uncased")  # Each page name must be tokenized first
model = DistilBertModel.from_pretrained("distilbert-base-uncased")  # The pages will then be run through an embedding layer and converted to vectors

TIMEOUT = 360  # time limit in seconds for the search

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/" # Constant string for the beginning of each link


def format_wikipedia_url(page_name): # Formats the user's input as an actual Wikipedia page
  return urljoin(WIKIPEDIA_BASE_URL, page_name)

def get_page_title(url):
  parsed_url = urlparse(url)
  page_title = unquote(parsed_url.path.split('/')[-1])
  page_title = page_title.replace('_', ' ')
  return page_title
  

def get_links(page_url):
  print(f"Fetching page: {get_page_title(page_url)}")
  response = requests.get(page_url)
  print(f"Finished fetching page: {get_page_title(page_url)}")

  soup = BeautifulSoup(response.text, 'html.parser')
  all_links = [urljoin(page_url, a['href']) for a in soup.find_all('a', href=True)if '#' not in a['href']]
  links = [link for link in all_links if 'en.wikipedia.org/wiki/' in link and '#' not in link]

  print(f"Found {len(links)} links on page: {get_page_title(page_url)}")
  return links


def embed_text(text):
  inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length") # Utilizes PyTorch tensors, which maps between vectors
  with torch.no_grad(): # Disables gradient calculation, which reduces memory consumption and therefore makes the program less computationally expensive
    outputs = model(**inputs)
  embeddings = outputs.last_hidden_state[:, 0, :].numpy()  
  return embeddings


def cosine_similarity(vector1, vector2):
  vector1 = vector1.flatten() # Flattens the vectors to one dimension
  vector2 = vector2.flatten()
  return (np.dot(vector1, vector2)) / (norm(vector1) * norm(vector2)) # Ranges between 0 and 1


def find_path(start_page, end_page):
    
    start_page_embedding = embed_text(start_page)
    end_page_embedding = embed_text(end_page)

    start_page_link = format_wikipedia_url(start_page)
    end_page_link = format_wikipedia_url(end_page)

    queue = [(start_page_link, [start_page_link], 0, start_page_embedding)]
    discovered = set()

    start_time = time.time()
    elapsed_time = time.time() - start_time

    while queue:
        
        queue.sort(key=lambda x: -cosine_similarity(x[3], end_page_embedding)) # Sorts the pages in the queue by cosine similarity 
        (vertex, path, depth, embedding) = queue.pop(0)

        if elapsed_time > TIMEOUT:
              raise TimeoutErrorWithLogs("Search exceeded time limit.", elapsed_time, len(discovered))
        
        for next in set(get_links(vertex)) - discovered:
            next_text = get_page_title(next)
            next_embedding = embed_text(next_text)
            
            if next == end_page_link:
                print(f"Found finish page: {next}")
                return path + [next]
            else:
                discovered.add(next)
                queue.append((next, path + [next], depth + 1, next_embedding))

    return []



class TimeoutErrorWithLogs(Exception):

  def __init__(self, message, time, discovered):
    super().__init__(message)
    self.time = time
    self.discovered = discovered


def main():
  
  print("\nWelcome to the Wikipedia Search Game! This search algorithm is powered by DistilBERT.\n")

  start_page = input("Please enter a start page: ")
  finish_page = input("Please enter an end page: ")

  shortest_path = find_path(start_page, finish_page)
  print(f"\nShortest path from {start_page} to {finish_page}:")
  print(" -> ".join([page.split('/')[-1].replace("_", " ").replace("wiki ", "") for page in shortest_path])) # Converts the pages into a readable format

if __name__ == "__main__":
  main()
