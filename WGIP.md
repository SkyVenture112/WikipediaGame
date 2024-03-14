# Wikipedia Game Improvement Proposal

# Description
There are a few aspects in which the breadth-first search algorithm for this program can be optimized:

* The algorithm can implement a set rather than a list to contain discovered pages. Sets are quite literally more efficient than lists regardless of what kind of operation is used. This would both allow for a
  a quicker overall runtime. Additionally, it would introduce an additional form of error checking considering that sets cannot contain duplicate values. If an additional page were somehow to be added twice or more
  to the discovered list, that could potentially disrupt the algorithm's accuracy and efficiency. It is therefore much more advisable to use a list in this instance.

  __Pseudo-Code Representation:__ ```
  create a new set</br>initialize it to having the start_page as its sole value  
                                  ```

*  The regular expression matching feature that is present within the get_links() function in crawler.py currently compiles the regular expression pattern each time get_links() is called. This is very computationally inefficient, as it wastes resources each time a new link is discovered (especially on pages with an immense amount of links). A potential solution to this could be the regular expression pattern being compiled once and then utilized each time a page must be scraped for links. This would reduce the overall time required to scrape not only the starting page, but every page traveled to after it.

    __Pseudo-Code Representation:__

   ```
pattern = compile_regular_expression(r'^https://en\.wikipedia\.org\wiki/[^:]*$')  

if the link matches the defined regular expression pattern and does not contain '#":  
  add the link to the set of filtered links
  ```
