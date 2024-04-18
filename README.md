# WikipediaGame

This program utilizes DistilBERT to optimize the search algorithm between a starting Wikipedia page and an ending Wikipedia page.

## Installation

(these instructions should work under GNU/Linux and Macos and WSL)

Prerequisites: Python

Please ensure the necessary libraries have been installed prior to running the program.

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

## Limitations

- Cannot be run in environments with minimal resources (i.e Replit)

## Further Ideas

- Potentially utilize most of a given page (rather than just its title) to optimize the search 
