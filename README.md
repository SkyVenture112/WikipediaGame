# WikipediaGame

This program utilizes DistilBERT to optimize the search algorithm between a starting Wikipedia page and an ending Wikipedia page. The process starts by collecting every link on the start page and embedding them via DistilBERT. The cosine similarity between each link and that of the end page is then calculated, and the link with the highest similarity is then chosen to follow. The cycle repeats itself until the end page is reached.

## Installation

Prerequisites: Python

Please ensure the following libraries have been installed prior to running the program:

```
pip install transformers
pip install requests
pip install torch
pip install numpy
pip install bs4
```

## Execution

```
git clone https://github.com/SkyVenture112/WikipediaGame
cd WikipediaGame/server
python crawler.py
```
## Testing

I used a variety of different start and end pages that ranged in similarity (i.e. "apple" and "banana" vs. "Undertale" and "Quantum physics"). What I discovered was that the further apart the pages are, the less likely the algorithm is going to be successful. Though any two Wikipedia pages can be used as the start and ending pages, using pages that are more closely related are going to provide for a quicker and more accurate result. Running the program locally with all of the libraries properly installed should allow the algorithm to run (albeit slowly). Replit was also used for testing purposes, though its cloud-based nature forced it to either run sluggishly or not at all.


## Limitations

- Cannot run efficiently in environments with minimal resources (i.e Replit)

## Further Ideas

- Potentially utilize most of a given page (rather than just its title) to optimize the search 
