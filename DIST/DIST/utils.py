import numpy as np

def minibatch(dataset0,dataset1,batch_size=128,seed=0):
    ## Get minibatch for two matched datasets.
    np.random.seed(seed)
    size = len(dataset1) 
    mini_batches0 = []
    mini_batches1 = []
    shuffle_dataset0 = [None] * size
    shuffle_dataset1 = [None] * size
    permutation = list(np.random.permutation(size))
    for i in range(size):
        shuffle_dataset0[i] = dataset0[permutation[i]]
        shuffle_dataset1[i] = dataset1[permutation[i]]
    for i in range(int(size//batch_size)+1):
        start_index = i*batch_size
        end_index = (i+1)*batch_size
        if end_index <= size:
            data0 = shuffle_dataset0[start_index:end_index]
            data1 = shuffle_dataset1[start_index:end_index]
        else:                                                   
            data0 = shuffle_dataset0[start_index:size]
            data0.extend(shuffle_dataset0[0:end_index-size])
            data1 = shuffle_dataset1[start_index:size]
            data1.extend(shuffle_dataset1[0:end_index-size])
            
        mini_batches0.append(data0)
        mini_batches1.append(data1)
    return mini_batches0, mini_batches1

def minibatch_list(dataset0,dataset1,mask,batch_size=128,seed=0):
    ## Get minibatch for three matched lists, which contain a list of matched datasets.
    np.random.seed(seed)
    mini_batches_list0=[]
    mini_batches_list1=[]
    mask_list=[]
    for i in range(len(dataset0)):
        mini_batches0, mini_batches1 = minibatch(dataset0[i],dataset1[i],batch_size=batch_size,seed=seed)
        mini_batches_list0.extend(mini_batches0)
        mini_batches_list1.extend(mini_batches1)
        mask_list.extend([mask[i]]*len(mini_batches0))
    
    size=len(mini_batches_list0)
    shuffle_mini_batches_list0 = [None] * size
    shuffle_mini_batches_list1 = [None] * size
    shuffle_mask_list = [None] * size
    permutation = list(np.random.permutation(size))
    for i in range(size):
        shuffle_mini_batches_list0[i] = mini_batches_list0[permutation[i]]
        shuffle_mini_batches_list1[i] = mini_batches_list1[permutation[i]]
        shuffle_mask_list[i] = mask_list[permutation[i]]
    return shuffle_mini_batches_list0, shuffle_mini_batches_list1, shuffle_mask_list