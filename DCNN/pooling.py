__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

import theano.tensor as T
from lasagne.layers.base import Layer
from utils import as_tuple
from lasagne.layers.pool import pool_output_length
from lasagne.layers.pool import pool_2d
class PoolPerLine(Layer):

    def __init__(self, incoming, pool_size, stride=None, pad=0,
                     ignore_border=True, mode='max', **kwargs):
            super(PoolPerLine, self).__init__(incoming, **kwargs)

            self.pool_size = as_tuple(pool_size, 1)
            self.stride = self.pool_size if stride is None else as_tuple(stride, 1)
            self.pad = as_tuple(pad, 1)
            self.ignore_border = ignore_border
            self.mode = mode
            

    def get_output_shape_for(self, input_shape):

        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[-1] = pool_output_length(input_shape[-1],
                                              pool_size=self.pool_size[0],
                                              stride=self.stride[0],
                                              pad=self.pad[0],
                                              ignore_border=self.ignore_border,
                                              )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        #Add one additional dimension in the rigth and give the 5d tensor for the
        #pooling. the 2d pooling will do the polling in the last 2 dimensiotns
        #the rows and the new additional one. then we kick it out in the return [:,:,:,:,0]
        input_5d = T.shape_padright(input, 1)
        pool=pool_2d(input_5d,
                         ws=(self.pool_size[0], 1),
                         stride=(self.stride[0], 1),
                         ignore_border=self.ignore_border,
                         pad=(self.pad[0], 0),
                         mode=self.mode,
                         )

        return pool[:,:, :, :, 0]


class KMaxPoolLayer(Layer):

    def __init__(self,incoming,k,**kwargs):
        super(KMaxPoolLayer, self).__init__(incoming, **kwargs)
        self.k = k

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.k)

    def get_output_for(self, input, **kwargs):
        return self.kmaxpooling(input,self.k)


    def kmaxpooling(self,input,k):

        #take the index of sorted values per chanel, sorted values from min to max
        sorted_values = T.argsort(input,axis=3)

        #take the k indexes from the end. These are the indexes of the top k values per layer(channel)per line
        topmax_indexes = sorted_values[:,:,:,-k:]

        # sort indexes so that we keep the correct order within the sentence
        topmax_indexes_sorted = T.sort(topmax_indexes)

        #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
        dim0 = T.arange(0,self.input_shape[0]).repeat(self.input_shape[1]*self.input_shape[2]*k)
        dim1 = T.arange(0,self.input_shape[1]).repeat(k*self.input_shape[2]).reshape((1,-1)).repeat(self.input_shape[0],axis=0).flatten()
        dim2 = T.arange(0,self.input_shape[2]).repeat(k).reshape((1,-1)).repeat(self.input_shape[0]*self.input_shape[1],axis=0).flatten()
        dim3 = topmax_indexes_sorted.flatten()
        return input[dim0,dim1,dim2,dim3].reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2], k))

    def kmaxpooling_numpy(input,k):
        input_shape=input.shape
        reshaped=input.reshape(input_shape[0],input_shape[1],1,-1)
        reshaped_shape=reshaped.shape

        sorted_values = np.argsort(reshaped,axis=3)
        topmax_indexes = sorted_values[:,:,:,-k:]
        topmax_indexes_sorted = np.sort(topmax_indexes)

        dim0 = np.arange(0,reshaped_shape[0]).repeat(reshaped_shape[1]*reshaped_shape[2]*k)
        dim1 = np.arange(0,reshaped_shape[1]).repeat(k*reshaped_shape[2]).reshape((1,-1)).repeat(reshaped_shape[0],axis=0).flatten()
        dim2 = np.arange(0,reshaped_shape[2]).repeat(k).reshape((1,-1)).repeat(reshaped_shape[0]*reshaped_shape[1],axis=0).flatten()
        dim3 = topmax_indexes_sorted.flatten()
        return reshaped[dim0,dim1,dim2,dim3].reshape((reshaped_shape[0], reshaped_shape[1], reshaped_shape[2], k))

    def kmaxpooling_matrix(self,input,k):
        input_shape=input.shape
        reshaped=input.reshape(input_shape[0],input_shape[1],1,-1)
        reshaped_shape=reshaped.shape

        #take the index of sorted values per chanel, sorted values from min to max
        sorted_values = T.argsort(reshaped,axis=3)

        #take the k indexes from the end. These are the indexes of the top k values per layer(channel)per line
        topmax_indexes = sorted_values[:,:,:,-k:]

        # sort indexes so that we keep the correct order within the sentence
        topmax_indexes_sorted = T.sort(topmax_indexes)

        #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
        dim0 = T.arange(0,self.input_shape[0]).repeat(self.input_shape[1]*self.input_shape[2]*k)
        dim1 = T.arange(0,self.input_shape[1]).repeat(k*self.input_shape[2]).reshape((1,-1)).repeat(self.input_shape[0],axis=0).flatten()
        dim2 = T.arange(0,self.input_shape[2]).repeat(k).reshape((1,-1)).repeat(self.input_shape[0]*self.input_shape[1],axis=0).flatten()
        dim3 = topmax_indexes_sorted.flatten()
        return reshaped[dim0,dim1,dim2,dim3].reshape((reshaped_shape[0], reshaped_shape[1], reshaped_shape[2], k))




class DynamicKMaxPoolLayer(KMaxPoolLayer):

    def __init__(self,incoming,ktop,nroflayers,layernr,**kwargs):
        super(DynamicKMaxPoolLayer, self).__init__(incoming,ktop, **kwargs)
        self.ktop = ktop
        self.layernr = layernr
        self.nroflayers = nroflayers

    def get_k(self,input_shape):
        """
        calculates k. k is the number of max values that will be pooled from a vector.
        """
        return T.cast(T.max([self.ktop,T.ceil((self.nroflayers-self.layernr)/float(self.nroflayers)*input_shape[3])]),'int32')

    def get_k_matrix_np(input_shape):


        return np.array([np.max([ktop,np.ceil((nroflayers-layernr)/float(nroflayers)*input_shape[2]*input_shape[3])])]).astype(int)

    def get_k_matrix(self,input_shape):
        """
        calculates k. k is the number of max values that will be pooled from a matrix.
        np.cast(np.max([ktop,np.ceil((nroflayers-layernr)/float(nroflayers)*input_shape[2]*input_shape[3])]),'int32')
        """
        return T.cast(T.max([self.ktop,T.ceil((self.nroflayers-self.layernr)/float(self.nroflayers)*input_shape[3])]),'int32')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], None)

    def get_output_for(self, input, **kwargs):

        k = self.get_k_matrix(input.shape)
        return self.kmaxpooling(input,k)


class DynamicKAreasMaxPoolLayer(Layer):

    def __init__(self,incoming,ktop,nroflayers,layernr,**kwargs):
        super(DynamicKMaxPoolLayer, self).__init__(incoming,ktop, **kwargs)
        self.ktop = ktop
        self.layernr = layernr
        self.nroflayers = nroflayers

    def get_k_matrix_np(input_shape):


        return np.array([np.max([ktop,np.ceil((nroflayers-layernr)/float(nroflayers)*input_shape[2]*input_shape[3])])]).astype(int)

    def get_k_matrix(self,input_shape):
        """
        calculates k. k is the number of max values that will be pooled from a matrix.
        np.cast(np.max([ktop,np.ceil((nroflayers-layernr)/float(nroflayers)*input_shape[2]*input_shape[3])]),'int32')
        """
        return T.cast(T.max([self.ktop,T.ceil((self.nroflayers-self.layernr)/float(self.nroflayers)*input_shape[3])]),'int32')


    def k_areas_maxpooling_np(input,k):

        #dynamic filter size
        f=int(np.ceil(np.sqrt((input.shape[-1]*input.shape[-2])/float(k))))

        #how many zero rows have to inserted to the end so that the size of the filter to fit exaclty to the number of rows.
        rows_to_insert=input.shape[-1]%f

        #how many zero columns have to inserted top the end so that the size of the filter to fit exaclty to the number of columns.
        columns_to_insert=input.shape[-2]%f

        #insert rows
        output=np.insert(input, input.shape[2]*np.ones(1).repeat(rows_to_insert), 0, axis=2)

        #insert columns
        output=np.insert(output, output.shape[3]*np.ones(1).repeat(columns_to_insert), 0, axis=3)

        output_shape=output.shape
        #take max out of every f (filter size) rows
        output=np.transpose(output, (0,1,3,2)).reshape(output_shape[0],output_shape[1],-1,f).max(3).reshape(output_shape[0],output_shape[1],output_shape[-1],-1).transpose((0,1,3,2))

        #take the max out of every f(filter size) columns
        output=output.reshape(output.shape[0],output.shape[1],-1,f).max(3).reshape(output_shape[0],output_shape[1],output_shape[2]/f,-1)

        return output

    def k_areas_maxpooling(input,k):

        #dynamic filter size
        f=int(T.ceil(T.sqrt((input.shape[-1]*input.shape[-2])/float(k))))

        #how many zero rows have to inserted to the end so that the size of the filter to fit exaclty to the number of rows.
        rows_to_insert=input.shape[-1]%f

        #how many zero columns have to inserted top the end so that the size of the filter to fit exaclty to the number of columns.
        columns_to_insert=input.shape[-2]%f

        #insert rows
        output=T.insert(input, input.shape[2]*T.ones(1).repeat(rows_to_insert), 0, axis=2)

        #insert columns
        output=T.insert(output, output.shape[3]*T.ones(1).repeat(columns_to_insert), 0, axis=3)

        output_shape=output.shape
        #take max out of every f (filter size) rows
        output=T.transpose(output, (0,1,3,2)).reshape(output_shape[0],output_shape[1],-1,f).max(3).reshape(output_shape[0],output_shape[1],output_shape[-1],-1).transpose((0,1,3,2))

        #take the max out of every f(filter size) columns
        output=output.reshape(output.shape[0],output.shape[1],-1,f).max(3).reshape(output_shape[0],output_shape[1],output_shape[2]/f,-1)

        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], None)

    def get_output_for(self, input, **kwargs):

        #k = self.get_k_matrix(input.shape)
        k=1
        return self.kmaxpooling_matrix(input,k)
