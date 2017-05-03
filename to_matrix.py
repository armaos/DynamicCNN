__author__ = 'Alexandros Armaos  (alexandros@tartaglialab.com )'


import numpy as np

aa='ACDEFGHIKLMNOPQRSTUV'
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

def two_hot(rseq,protseq):


    m=np.zeros((len(a),len(b),24),dtype=np.bool_)
    for i in range(len(a)):
        for j in range(len(b)):
            m[i][j]=np.append(one_hot_nt[b[j]],one_hot_aa[a[i]])

    #m shape: numofchanels, len_rna, len_protein
    return np.transpose(m,(2,0,1))
