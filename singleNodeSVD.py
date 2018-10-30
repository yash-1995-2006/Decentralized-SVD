import numpy as np
import time

class singleNodeSVD():
    def __init__(self, initial_data, learning_rate, k_rank, shared_queue, iterations):
        '''
        Initialize the class with parameters
        :param initial_data: Initial shard of data at the given node
        :param learning_rate: learning rate for algorithm
        :param k_rank: rank of approximated matrices
        :param shared_queue: queue shared between threads to exchange Y
        :param iterations: number of iterations to convergence
        '''
        self.A = initial_data
        self.learning_rate = learning_rate
        self.k = k_rank
        self.iterations = iterations
        self.m = self.A.shape[0]
        self.n = self.A.shape[1]
        self.initialize_XY()
        self.error = 0
        self.error_history = []
        self.lock = 0
        self.q = shared_queue

    def initialize_XY(self, low=0, high=1):
        '''
        Initialize the values of X and Y with the given parameters
        :param low: Lower limit of uniform distribution
        :param high: Upper limit of uniform distribution
        :return:
        '''
        self.X = np.random.uniform(low=low, high=high, size=[self.m, self.k])
        self.Y = np.random.uniform(low=low, high=high, size=[self.n, self.k])

    def update_Y(self, new_Y):
        '''
        set the new value of Y
        :param new_Y: the new metrix to replace Y
        :return:
        '''
        while self.lock==1:
            time.sleep(secs=5)
        self.Y = new_Y

    def share_Y(self):
        '''
        Send the new value of y to a queue being shared by the threads
        :return:
        '''
        self.q.append(self.Y)

    def run_iterations(self, iterations):
        '''
        Run the optmization for a set number of iterations
        :param iterations: number of optimization iterations
        :return:
        '''
        for i in range(iterations):
            Ai = self.A
            for l in range(self.k):
                xl = np.reshape(self.X[:,l],[-1,1])
                yl = np.reshape(self.Y[:,l],[-1,1])
                err = Ai - np.matmul(xl, np.transpose(yl))
                xl_new = xl + (self.learning_rate * np.matmul(err, yl))
                yl_new = yl + (self.learning_rate * np.matmul(np.transpose(err), xl))
                self.X[:,l] = np.reshape(xl_new, [-1])
                self.Y[:,l] = np.reshape(yl_new, [-1])
                Ai = Ai - np.matmul(xl_new,np.transpose(yl_new))
            if i % 50:
                p = np.random.uniform(low=0, high=1)
                if p > 0.7:
                    self.share_Y()
                elif p < 0.2 and len(self.q) > 0:
                    self.Y = self.q.pop(0)

    def runSVD(self):
        '''
        Driver code for testing the above code
        :return:
        '''
        self.run_iterations(self.iterations)
        print("Original Matrix:\n", self.A)
        print("\nMult of decomposed arrays:\n", np.matmul(self.X, np.transpose(self.Y)).astype("float16"))

# A = np.reshape(range(50), [5,10]).astype("float32")
# svd = singleNodeSVD(initial_data=A, learning_rate=0.001, k_rank=2)
# svd.runSVD()



