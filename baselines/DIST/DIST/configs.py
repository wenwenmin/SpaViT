class Config:
    # network meta params
    epoch = 200
    batch_size = 128
    width = 64
    iteration_depth = 5 # layers number of iteration, range from conv layers
    iteration = 2 # times of iteration
    learning_rate = 0.001
    init_variance = 0.1 # variance of weight initializations, typically smaller when residual learning is on   
    test_positive = True # True means positive data, outputs are clipped by zeros.

    def __init__(self):
        # network meta params that by default are determined (by other params) by other params but can be changed
        self.filter_shape = ([[3, 3, 1, self.width]] +
                             [[3, 3, self.width, self.width]] * (self.iteration_depth * self.iteration - 1) +
                             [[3, 3, self.width, 4]])
        self.depth = self.iteration_depth * self.iteration + 1
