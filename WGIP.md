# Wikipedia Game Improvement Proposal
Jack Dippel

# Description
Here is an aspect in which the breadth-first search algorithm for this program could be optimized:

* The algorithm can implement an open-source natural language processing model such as Google's BERT. This would allow the algorithm to more thoroughly comprehend the semantics of the words it processes and therefore devise a path to the final page much more quickly. BERT, in particular, is a bi-directional model, meaning it uses every word in a page or link to grasp its overall meaning. It is specifically trained to understand semantics and sentiment, and can also generate word embeddings to further augment the algorithm's efficiency. Links would be able to be categorized into distinct groups, thereby reducing the amount of time wasted on traversing paths that ultimately lead to completely unrelated pages.
  
  __Pseudo-Code Representation:__


        def embed_text(text):
           use the NLP model to embed the text
           return the embedding

        def calculate_similarity(embedding1, embedding2):
           use the NLP model to calculate the similarity between two embeddings
           return the degree of similarity // This would be used to choose which links to travel to later in the search

        for each link in the current page:
            if page has not been visited:
                embed the link
                calculate the similarity of the page and the current page
                append page to the discovered list
            else if the page is the destination page:
                return the page

  # Milestones
  * 3/28 - Ensure that all necessary packages and libraries have been installed or imported (i.e. the transformers Python library, PyTorch, numpy, etc.). This will allow the development environment to actually implement BERT's functions and capabilities. **(Complete)**

  * 4/4 - Choose a pre-trained BERT model from HuggingFace and load it into the algorithm. For the sake of simplicity and error avoidance, the program will (currently) not fine-tune BERT on a dataset. This may change if BERT does not optimize the algorithm as effectively as it would were it fine-tuned on a dataset of Wikipedia pages. **(Complete)**
    
  * 4/16 - Implement BERT's tokenization and masking feature to embed the input text. This would allow for the algorithm to be "guided" based on the content of each Wikipedia page. Additionally, utilize cosine similarity in Python to calculate the degree of similarity between the now-embedded vectors. **(Complete)**
    
  * 4/18 - Finish completely integrating BERT's functions with the algorithm and clean up any code as necessary. **(Complete)**
  

       

        
  

   
       
  
