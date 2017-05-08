__author__ = 'Alexandros Armaos  (alexandros@tartaglialab.com )'

import numpy as np
import IPython
import collections
import matplotlib.pyplot as plt
aa='ARNDCQEGHIKLMFPSTWYV'

nt="ACGT"
one_hot_nt={}
for i, l in enumerate(nt):
    bits = np.zeros((1)).repeat(4); bits[i] = '1'
    one_hot_nt[l] = bits
    #print one_hot

one_hot_aa={}
for i, l in enumerate(aa):
    bits = np.zeros((1)).repeat(20); bits[i] = '1'
    one_hot_aa[l] = bits
#this is for the non-aa..for padding
one_hot_aa['X']=np.zeros((1)).repeat(20)


def one_hot(protseq):


    m=np.zeros((len(protseq),20),dtype=np.int8)
    for i in range(len(protseq)):
        m[i]=one_hot_aa[protseq[i]]

    #m shape: numofchanels, len_rna, len_protein
    return np.transpose(m,(1,0))

def two_hot(rseq,protseq):


    m=np.zeros((len(rseq),len(protseq),24),dtype=np.bool_)
    for i in range(len(protseq)):
        for j in range(len(rseq)):
            m[i][j]=np.append(one_hot_nt[rseq[j]],one_hot_aa[protseq[i]])

    #m shape: numofchanels, len_rna, len_protein
    return np.transpose(m,(2,0,1))

def read_and_sort_matlab_data(x_file,y_file,padding_value=15448):


    sorted_dict = {}
    x_data = []
    i=0
    file = open(x_file,"r")
    for line in file:
        words = line.split(",")
        result = []
        length=None
        for word in words:
            word_i = int(word)
            if word_i == padding_value and length==None:
                length = len(result)
            result.append(word_i)
        x_data.append(result)

        if length==None:
            length=len(result)

        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i+=1

    file.close()

    file = open(y_file,"r")
    y_data = []
    for line in file:
        words = line.split(",")
        y_data.append(int(words[0])-1)
    file.close()

    new_train_list = []
    new_label_list = []
    lengths = []
    for length, indexes in sorted_dict.items():
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)

    return np.asarray(new_train_list,dtype=np.int32),np.asarray(new_label_list,dtype=np.int32),lengths

def read_data_1d(x_file,y_file,padding_value=100):

    print "loading features"
    sorted_dict = {}
    x_data = []
    i=0
    file=open(x_file,"r")

    N=50000
    #for line in file:

    for k in range(N):
        line=file.next().strip()
        #line=line.strip()
        seq=line+''.join(['X' for j in range(len(line),100)])
        x_data.append(one_hot(seq))
        length=len(line)
        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i+=1

    file.close()
    file = open(y_file,"r")



    y_data = []
    print "loading labels"
    for k in range(N):
        line=file.next().strip()
    #for line in file:
        #line=line.strip()
        y_data.append(int(line.strip()))
        """if line=='0':
            y_data.append(np.array([0,1], dtype=np.int8))
        elif line=='1':
            y_data.append(np.array([1,0], dtype=np.int8))"""


    file.close()


    new_train_list = []
    new_label_list = []
    lengths = []
    print "building new lists"
    for length, indexes in sorted_dict.items():
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)


    return np.asarray(new_train_list,dtype=np.int32),np.asarray(new_label_list,dtype=np.int32),lengths


def pad_to_batch_size(array,batch_size):

    rows_extra = batch_size - (array.shape[0] % batch_size)
    if len(array.shape)==1:
        padding = np.zeros((rows_extra,),dtype=np.int32)
        return np.concatenate((array,padding))

    elif len(array.shape)==2:
        padding = np.zeros((rows_extra,array.shape[1]),dtype=np.int32)
        return np.vstack((array,padding))

    elif len(array.shape)==3:
        padding = np.zeros((rows_extra,array.shape[1],array.shape[2]),dtype=np.int32)

    elif len(array.shape)==4:
        padding = np.zeros((rows_extra,array.shape[1],array.shape[2],array.shape[3]),dtype=np.int32)

        return np.vstack((array,padding))

def extend_lenghts(length_list,batch_size):
    elements_extra = batch_size - (len(length_list) % batch_size)
    length_list.extend([length_list[-1]]*elements_extra)

def check_plots(title,train_costs,validation_accuraces, testing_accuraces_accuraces):
    folder="data/figures/"+title
    fig = plt.figure()
    plt.plot(range(len(train_costs)),train_costs)
    plt.legend("training cost", loc='upper left')
    fig.savefig(folder+'-train_costs.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(range(len(validation_accuraces)),validation_accuraces)
    plt.plot(range(len(testing_accuraces_accuraces)),testing_accuraces_accuraces)
    plt.legend(['Validation acc', 'Test acc'], loc='upper left')
    fig.savefig(folder+'-Accuracies.png', dpi=fig.dpi)
