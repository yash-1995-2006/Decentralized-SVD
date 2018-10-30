# Rank-K Approximation of a matrix using Decentralized SVD

This repository cinsists of code that calculates the rank-k approximation of a matrix using singular value decomposition on a decentralized system. The file `singleNodeSVD.py` consists of the code that runs on one node where as `Multinode Simulation.py` simulates multiple nodes on one system by running multiple threads concurrently. The above code uses the Gossip Protocol to share data between nodes. This sharing is simulated by using a queue that is shared by all the threads.

To run the code, run the following command,

```python Multinode\ Simulation.py```

The simulator crates its own data but you can add your own data for calculating the low-rank matrix. You can change the other parameters in the simulation arguements.