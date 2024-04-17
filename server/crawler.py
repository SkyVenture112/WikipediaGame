import time
import requests
from bs4 import BeautifulSoup
import re
from transformers import DistilBertTokenizer, DistilBertModel  # I used DistiLBERT rather than BERT because the latter was too computationally expensive
from urllib.parse import quote, urljoin
import numpy as np  # These two libraries are used to calculate the cosine similarities between two vectors
from numpy.linalg import norm
import torch
import torch.nn.functional as F

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased")  # Each page name must be tokenized first
model = DistilBertModel.from_pretrained(
    "distilbert-base-uncased"
)  # The pages will then be run through an embedding layer and converted to vectors

TIMEOUT = 60  # time limit in seconds for the search

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"


def format_wikipedia_url(
    page_name):  # Formats the user's input as an actual Wikipedia page
  return urljoin(WIKIPEDIA_BASE_URL, page_name)


def get_links(page_url):
  print(f"Fetching page: {page_url}")
  response = requests.get(page_url)
  print(f"Finished fetching page: {page_url}")

  soup = BeautifulSoup(response.text, 'html.parser')
  all_links = [
      urljoin(page_url, a['href']) for a in soup.find_all('a', href=True)
      if '#' not in a['href']
  ]
  links = [
      link for link in all_links
      if 'en.wikipedia.org/wiki/' in link and '#' not in link
  ]

  print(f"Found {len(links)} links on page: {page_url}")
  return links


def embed_text(text):
  inputs = tokenizer(text,
                     return_tensors="pt",
                     max_length=512,
                     truncation=True,
                     padding="max_length")
  with torch.no_grad():
    outputs = model(**inputs)
  embeddings = outputs.last_hidden_state[:, 0, :].numpy(
  )  # Taking the [CLS] token's embedding
  return embeddings


def cosine_similarity(vector1, vector2):
  return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def find_most_similar_link(queue, end_page_embedding):
  best_link = None
  best_similarity = -1
  for link, path, depth, embedding in queue:
    similarity = cosine_similarity(embedding, end_page_embedding)
    if similarity > best_similarity:
      best_similarity = similarity
      best_link = (link, path, depth)
  return best_link


def find_path(start_page, finish_page):
  start_url = format_wikipedia_url(start_page)
  finish_url = format_wikipedia_url(finish_page)

  start_embedding = embed_text(start_page)
  finish_embedding = embed_text(finish_page)

  queue = [(start_url, [start_url], 0, start_embedding)]
  discovered = set()
  logs = []

  start_time = time.time()
  elapsed_time = time.time() - start_time
  while queue and elapsed_time < TIMEOUT:
    vertex, path, depth, vertex_embedding = queue.pop(0)
    for next_url in set(get_links(vertex)) - discovered:
      if next_url == finish_url:
        log = f"Found finish page: {next_url}"
        print(log)
        logs.append(log)
        logs.append(f"Search took {elapsed_time} seconds.")
        print(f"Search took {elapsed_time} seconds.")
        logs.append(f"Discovered pages: {len(discovered)}")
        return path + [next_url], logs, elapsed_time, len(discovered)
      else:
        log = f"Adding link to queue: {next_url} (depth {depth})"
        print(log)
        logs.append(log)

        next_embedding = embed_text(next_url)
        discovered.add(next_url)
        queue.append((next_url, path + [next_url], depth + 1, next_embedding))

    # Select the best link based on cosine similarity
    most_similar_link = find_most_similar_link(queue, finish_embedding)
    if most_similar_link:
      queue.remove(most_similar_link)
      queue.insert(0, most_similar_link)

    elapsed_time = time.time() - start_time

  logs.append(f"Search took {elapsed_time} seconds.")
  print(f"Search took {elapsed_time} seconds.")
  logs.append(f"Discovered pages: {len(discovered)}")
  raise TimeoutErrorWithLogs("The search has exceeded the time limit.", logs,
                             elapsed_time, len(discovered))


class TimeoutErrorWithLogs(Exception):

  def __init__(self, message, logs, time, discovered):
    super().__init__(message)
    self.logs = logs
    self.time = time
    self.discovered = discovered


def main():
  print(
      "\nWelcome to the Wikipedia Search Game! This search algorithm is powered by DistilBERT.\n"
  )
  start_page = input("Please enter a start page: ")
  finish_page = input("Please enter an end page: ")

  try:
    shortest_path, logs, elapsed_time, discovered_count = find_path(
        start_page, finish_page)
    print(f"\nShortest path from {start_page} to {finish_page}:")
    print(" -> ".join([
        page.split('/')[-1].replace("_", " ").replace("wiki ", "")
        for page in shortest_path
    ]))
  except TimeoutErrorWithLogs:
    print(TimeoutErrorWithLogs)


if __name__ == "__main__":
  main()
