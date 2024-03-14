# Wikipedia Game Improvement Proposal

# Description
There are a few aspects in which the breadth-first search algorithm for this program can be optimized:

* The algorithm can implement a set rather than a list to contain discovered pages. Sets are quite literally more efficient than lists regardless of what kind of operation is used. This would both allow for a
  a quicker overall runtime. Additionally, it would introduce an additional form of error checking considering that sets cannot contain duplicate values. If an additional page were somehow to be added twice or more
  to the discovered list, that could potentially disrupt the algorithm's accuracy and efficiency. It is therefore much more advisable to use a list in this instance.

  __Pseudo-Code Representation:__ ```
  discovered = set([start_page])
                                  ```

*  
