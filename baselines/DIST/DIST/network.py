import tensorflow as tf
from .configs import Config
from .utils import *
import time


class Net: 
    train_lr = []
    train_hr = []
    train_in_tissue = []

    def __init__(self, train_set, test_set, conf=Config()):
        
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = conf
        
        # Read input and output of training set(numpy array or a list of numpy arrays)
        if type(train_set[0])==list:
            for i in range(len(train_set)):
                self.train_lr.append(train_set[i][0])
                self.train_hr.append(train_set[i][1])
                self.train_in_tissue.append(train_set[i][2])
        else:
            self.train_lr.append(train_set[0])
            self.train_hr.append(train_set[1])
            self.train_in_tissue.append(train_set[2])
        
        # Read input of testing set(numpy array)
        self.test_set = test_set
        
        # Prepare TF default computational graph
        self.model = tf.Graph()

        # Build network computational graph
        self.build_network(conf)
            
        # Initialize network weights and meta parameters
        self.init_sess()

    def run(self):
        
        # Training
        start = time.perf_counter()
        t = self.conf.epoch
        for epoch in range(t):
            # Get minibatch
            self.train_lr_batch, self.train_hr_batch,self.train_in_tissue_batch = minibatch_list(self.train_lr,self.train_hr,self.train_in_tissue,batch_size=self.conf.batch_size,seed=None)
            
            # Main training process
            for self.input in zip(self.train_lr_batch, self.train_hr_batch,self.train_in_tissue_batch): 
                self.train()
            
            # Progress bar
            i = epoch + 1
            finish = "#" * round((i/t)*50)
            need_do = "-" * (50-round((i/t)*50))
            progress = (i / t) * 100
            dur = time.perf_counter() - start
            print("\rTraining: {:^3.0f}%[{}->{}]{:.2f}s".format(progress, finish, need_do, dur), end="")
        print("\n")
        
        # Testing
        start = time.perf_counter()
        t = len(self.test_set)
        test_output = np.zeros([t,self.test_set[0].shape[0]*2,self.test_set[0].shape[1]*2])
        for g,self.input in enumerate(self.test_set):
            # Testing process
            output = np.squeeze(self.forward_pass(np.expand_dims(self.input,-1)))
            test_output[g] = output
            
            # Progress bar
            i = g + 1
            finish = "#" * round((i/t)*50)
            need_do = "-" * (50-round((i/t)*50))
            progress = (i / t) * 100
            dur = time.perf_counter() - start
            print("\rTesting: {:^3.0f}%[{}->{}]{:.2f}s".format(progress, finish, need_do, dur), end="")
        print("\n")
        
        return test_output
    
    def build_network(self, meta):
        with self.model.as_default():

            # Learning rate tensor
            self.learning_rate_t = tf.placeholder(tf.float32, name='learning_rate')

            # Input image
            self.lr_t = tf.placeholder(tf.float32, name='lr')
                      
            # Ground truth
            self.hr_t = tf.placeholder(tf.float32, name='hr')
            
            # In tissue matrix
            self.in_tissue_t = tf.placeholder(tf.float32, name='in_tissue')
            
            # Filters
            self.filters_t = [tf.get_variable(shape=meta.filter_shape[ind], name='filter_%d' % ind, dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(
                                           stddev=np.sqrt(meta.init_variance/np.prod(
                                                     meta.filter_shape[ind][0:3]))))
                              for ind in range(meta.depth)]
            
            # Bias
            self.bias_t = [tf.get_variable(shape=meta.filter_shape[ind][-1], name='bias_%d' % ind, dtype=tf.float32,                                                     initializer=tf.zeros_initializer()) for ind in range(meta.depth)]

            # Activate filters on layers one by one (this is just building the graph, no calculation is done here)
            self.layers_t = [self.lr_t] + [None] * meta.depth
            for l in range(meta.depth - 1):
                if (l+1) % meta.iteration_depth == 1:
                    self.layers_t[l + 1] = tf.nn.bias_add(tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                       [1,1,1,1], "SAME", name='conv2d_%d'%(l+1)),self.bias_t[l], name='bias_%d'%(l+1))
                    self.layers_t[l + 2] = tf.nn.relu(self.layers_t[l + 1])
                if (l+1) % meta.iteration_depth > 2:
                    self.layers_t[l + 1] = tf.nn.bias_add(tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                       [1,1,1,1], "SAME", name='conv2d_%d'%(l+1)),self.bias_t[l], name='bias_%d'%(l+1))
                if (l+1) % meta.iteration_depth == 0:
                    self.layers_t[l + 1] = tf.nn.relu(self.layers_t[l] + self.layers_t[l - meta.iteration_depth + 2])
            
            # Last conv layer
            l = meta.depth - 1
            self.layers_t[l + 1] = tf.nn.bias_add(tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                       [1,1,1,1], "SAME", name='conv2d_%d'%(l+1)),self.bias_t[l], name='bias_%d'%(l+1))
              
            # Skip conncection to learn residuals
            self.output_t = self.layers_t[meta.depth] + tf.concat([self.lr_t,self.lr_t,self.lr_t,self.lr_t],axis=-1)

            # Final loss
            self.loss_t = tf.reduce_mean(tf.concat(
                [tf.reshape(tf.square((self.output_t[:,:,:,0]-tf.squeeze(self.hr_t[:,0::2,0::2]))*self.in_tissue_t[:,0::2,0::2]),[-1]),
                tf.reshape(tf.square((self.output_t[:,:,:,1]-tf.squeeze(self.hr_t[:,0::2,1::2]))*self.in_tissue_t[:,0::2,1::2]),[-1]),
                tf.reshape(tf.square((self.output_t[:,:,:,2]-tf.squeeze(self.hr_t[:,1::2,0::2]))*self.in_tissue_t[:,1::2,0::2]),[-1]),
                tf.reshape(tf.square((self.output_t[:,:,:,3]-tf.squeeze(self.hr_t[:,1::2,1::2]))*self.in_tissue_t[:,1::2,1::2]),[-1])],0))
            
            # Apply adam optimizer
            self.var1_t = tf.trainable_variables()[:meta.depth-1]+tf.trainable_variables()[meta.depth:-1]
            self.var2_t = tf.trainable_variables()[meta.depth-1:meta.depth*2:meta.depth]       
            self.train_op1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t).minimize(self.loss_t, var_list=self.var1_t)
            self.train_op2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t*0.001).minimize(self.loss_t, var_list=self.var2_t)
            self.train_op = tf.group(self.train_op1, self.train_op2)
            
            # Initialization
            self.init_op = tf.global_variables_initializer()

    def init_sess(self):
        
        # These are for GPU consumption, preventing TF to catch all available GPUs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Initialize computational graph session
        self.sess = tf.Session(graph=self.model, config=config)

        # Initialize weights
        self.sess.run(self.init_op)

        # Initialize parameters
        self.loss = []
        self.learning_rate = self.conf.learning_rate
        
    def forward_pass(self, lr):
        
        # Create feed dict
        feed_dict = {'lr:0': np.expand_dims(lr, 0)}
 
        # Run network
        output = np.squeeze(self.sess.run([self.output_t], feed_dict), axis=0)
        if self.conf.test_positive:
            output[output<0]=0
        up_output=np.zeros([output.shape[0]*2,output.shape[1]*2])
        up_output[0::2,0::2]=output[:,:,0]
        up_output[0::2,1::2]=output[:,:,1]
        up_output[1::2,0::2]=output[:,:,2]
        up_output[1::2,1::2]=output[:,:,3]
        return up_output

    def train(self):
        
        # Main training process
        self.lr = np.array(self.input[0])
        self.hr = np.array(self.input[1])
        self.in_tissue = np.array(self.input[2])
                
        # Create feed dict
        feed_dict = {'learning_rate:0': self.learning_rate,
                 'lr:0': np.expand_dims(self.lr,-1),
                 'hr:0': self.hr,
                 'in_tissue:0': np.expand_dims(self.in_tissue,0)}
             
        # Run network
        _,loss=self.sess.run([self.train_op,self.loss_t],feed_dict)
        self.loss.append(loss)