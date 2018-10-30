from singleNodeSVD import singleNodeSVD
import threading
import numpy as np

class simulation():
    def __init__(self, data, number_nodes, learning_rate, iterations, k_rank):
        '''
        Simulation Parameters
        :param data: original complete matrix of data
        :param number_nodes: number of nodes to simulate
        :param learning_rate: learning rate of algorithm
        :param iterations: number of iterations to perform
        :param k_rank: required rank of decomposed matrices
        '''
        self.q = []
        self.data = data
        self.number_nodes = number_nodes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.k = k_rank
        self.shard_data()

    def shard_data(self):
        '''
        shard data for each of the nodes
        :return:
        '''
        shard_size = self.data.shape[0] // self.number_nodes
        self.shards = []
        for i in range(self.number_nodes):
            self.shards.append(self.data[i*shard_size : (i+1)*shard_size])

    def run(self):
        '''
        Start simulation of the
        :return:
        '''
        threads = []
        for i in range(self.number_nodes):
            threads.append(threading.Thread(target=singleNodeSVD(initial_data=self.shards[i], learning_rate=self.learning_rate, k_rank=self.k, shared_queue=self.q, iterations=self.iterations).runSVD))
        for i in range(self.number_nodes):
            threads[i].start()


#A = np.random.normal(loc=0, scale=2, size=[500,100])
A = np.reshape(range(500), [50,10]).astype("float32")
s = simulation(data=A, number_nodes=4, learning_rate=0.0001, iterations=10000, k_rank=3)
s.run()