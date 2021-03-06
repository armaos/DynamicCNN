__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

from lasagne import *
from lasagne.layers import Layer
import lasagne.utils
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv2d
import IPython
import theano

# Adapted from Lasagne
class Conv1DLayer(Layer):

#Glorot: This is also known as Xavier initialization
#GlorotUniform: This is also known as Xavier initialization picking from Unifrom distribution
#GlorotNormal: This is also known as Xavier initialization picking from Unifrom distribution
    def __init__(self, incoming, num_filters, filter_size,
                 border_mode="valid",
                 W=lasagne.init.GlorotNormal(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Conv1DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = lasagne.utils.as_tuple(1, 1)

        if border_mode=="same":
            self.border_mode=self.filter_size-1
        else:
            self.border_mode = border_mode

        print self.input_shape

        self.num_input_channels = self.input_shape[1]
        self.num_of_rows = self.input_shape[2]

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            bias_temp_shape = self.get_output_shape_for(self.input_shape)
            biases_shape = (bias_temp_shape[1],bias_temp_shape[2])
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        return (self.num_filters,self.num_input_channels, self.num_of_rows, self.filter_size)

    def get_output_shape_for(self, input_shape):



        output_length = lasagne.layers.conv.conv_output_length(input_shape[-1],
                                           self.filter_size,
                                           self.stride[0],
                                           self.border_mode)

        return (input_shape[0], self.num_filters, self.num_of_rows, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):

        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = T.nnet.conv.conv2d(
                           input=input,
                           filters=self.W,
                           image_shape=input_shape,
                           filter_shape=filter_shape,
                           border_mode=self.border_mode,
                           )

        elif self.border_mode == 'same':
            raise NotImplementedError("Not implemented yet ")
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)


        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x',0,1,'x')


        return self.nonlinearity(activation)
# Adapted from Lasagne
class Conv2DLayer(Layer):

    def __init__(self, incoming, num_filters, filter_size,
                 border_mode="valid",
                 W=lasagne.init.GlorotNormal(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Conv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = lasagne.utils.as_tuple(1, 1)
        self.border_mode = border_mode


        # If it is an image the input shape will be 3
        # If it is a stack of filter ouputs after a previous convolution, the input shape will be 4
        if len(self.input_shape)==3:
            self.num_input_channels = 1
            self.num_of_rows = self.input_shape[1]
        elif len(self.input_shape)==4:
            self.num_input_channels = self.input_shape[1]
            self.num_of_rows = self.input_shape[2]

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            bias_temp_shape = self.get_output_shape_for(self.input_shape)
            biases_shape = (bias_temp_shape[1],bias_temp_shape[2],bias_temp_shape[3])
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        return (self.num_filters,self.num_input_channels, self.filter_size, self.filter_size)

    def get_output_shape_for(self, input_shape):

        output_height = lasagne.layers.conv.conv_output_length(input_shape[-2],
                                           self.filter_size,
                                           self.stride[1],
                                           self.border_mode)

        output_length = lasagne.layers.conv.conv_output_length(input_shape[-1],
                                           self.filter_size,
                                           self.stride[0],
                                           self.border_mode)

        return (input_shape[0], self.num_filters, output_height, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):

        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = T.nnet.conv.conv2d(
                           input=input,
                           filters=self.W,
                           image_shape=input_shape,
                           filter_shape=filter_shape,
                           border_mode=self.border_mode,
                           )
        elif self.border_mode == 'same':
            raise NotImplementedError("Not implemented yet ")
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)


        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x',0,1,'x')


        return self.nonlinearity(activation)

class Conv1DLayerSplittedSameFilter(Layer):
    """
        for this case we do convolution in all rows of the input using the same filter.
        I will use that in the case of sequence embeding that produces 3 rows of embeding accordint to the 3-gram (3mer of AAs)
        and the convolution will run on the 3 rows with applying the same filter.
    """
    def __init__(self, incoming, num_filters, filter_size,
                 border_mode="valid",
                 W=lasagne.init.GlorotNormal(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Conv1DLayerSplittedSameFilter, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = lasagne.utils.as_tuple(1, 1)

        self.border_mode = border_mode


        print "input shape", self.input_shape
        if len(self.input_shape)==3:
            self.num_input_channels = 1
            self.num_of_rows = self.input_shape[1]
        elif len(self.input_shape)==4:
            self.num_input_channels = self.input_shape[1]
            self.num_of_rows = self.input_shape[2]


        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:

            bias_temp_shape = self.get_output_shape_for(self.input_shape)
            biases_shape = (bias_temp_shape[1],bias_temp_shape[2])
            print "bias shape:" ,biases_shape
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        # get_W_shape to be with 1 row . this one row will apply to all rows of input
        return (self.num_filters,self.num_input_channels, self.num_of_rows, self.filter_size)

    def get_output_shape_for(self, input_shape):


        output_length = lasagne.layers.conv.conv_output_length(input_shape[-1],
                                           self.filter_size,
                                           self.stride[0],
                                           self.border_mode)

        print "output shape: ",(input_shape[0], self.num_filters, self.num_of_rows, output_length)
        return (input_shape[0], self.num_filters, self.num_of_rows, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):

        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        # We split the input shape and the filters into seperate rows to be able to execute a row wise 1D convolutions
        # BUT we will use the same filter for all rows.
        # We cannot convolve over the columns
        # However, we do need to convolve over multiple channels=output filters previous layer

        print "input_shape", self.input_shape

        if len(self.input_shape)==3:
            input_shape_row= (self.input_shape[0], 1, 1, self.input_shape[2])
            new_input = input.dimshuffle(0,'x', 1, 2)
        elif len(self.input_shape)==4:
            input_shape_row= (self.input_shape[0], self.input_shape[1], 1,  self.input_shape[3])
            new_input = input

        #
        filter_shape_row =(filter_shape[0],filter_shape[1],1,filter_shape[3])
        conveds = []

        #Note that this for loop is only to construct the Theano graph and will never be part of the computation
        print '\n'
        print "input shape: " ,input_shape
        print "filter shape:", filter_shape


        for i in range(self.num_of_rows):


            conveds.append(T.nnet.conv.conv2d(new_input[:,:,i,:].dimshuffle(0,1,'x',2),
                           self.W[:,:,i,:].dimshuffle(0,1,'x',2),
                           image_shape=input_shape_row,
                           filter_shape=filter_shape_row,
                           border_mode=self.border_mode,
                           ))

        conved = T.concatenate(conveds,axis=2)




        #IPython.embed()
        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x',0,1,'x')

        return self.nonlinearity(activation)

class Conv1DLayerSplitted(Layer):

    def __init__(self, incoming, num_filters, filter_size,
                 border_mode="valid",
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Conv1DLayerSplitted, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = lasagne.utils.as_tuple(1, 1)
        self.border_mode = border_mode


        # If it is an image the input shape will be 3
        # If it is a stack of filter ouputs after a previous convolution, the input shape will be 4
        print len(self.input_shape)
        if len(self.input_shape)==3:
            self.num_input_channels = 1
            self.num_of_rows = self.input_shape[1]
        elif len(self.input_shape)==4:
            self.num_input_channels = self.input_shape[1]
            self.num_of_rows = self.input_shape[2]

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            bias_temp_shape = self.get_output_shape_for(self.input_shape)
            biases_shape = (bias_temp_shape[1],bias_temp_shape[2])
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        return (self.num_filters,self.num_input_channels, self.num_of_rows, self.filter_size)

    def get_output_shape_for(self, input_shape):

        output_length = lasagne.layers.conv.conv_output_length(input_shape[-1],
                                           self.filter_size,
                                           self.stride[0],
                                           self.border_mode)

        return (input_shape[0], self.num_filters, self.num_of_rows, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):

        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()



        # We split the input shape and the filters into seperate rows to be able to execute a row wise 1D convolutions
        # We cannot convolve over the columns
        # However, we do need to convolve over multiple channels=output filters previous layer
        # See paper of Kalchbrenner for more details
        if self.border_mode in ['valid', 'full']:

            if len(self.input_shape)==3:
                input_shape_row= (self.input_shape[0], 1, 1, self.input_shape[2])
                new_input = input.dimshuffle(0,'x', 1, 2)
            elif len(self.input_shape)==4:
                input_shape_row= (self.input_shape[0], self.input_shape[1], 1,  self.input_shape[3])
                new_input = input

            filter_shape_row =(filter_shape[0],filter_shape[1],1,filter_shape[3])
            conveds = []

            #Note that this for loop is only to construct the Theano graph and will never be part of the computation
            for i in range(self.num_of_rows):
                conveds.append(T.nnet.conv.conv2d(new_input[:,:,i,:].dimshuffle(0,1,'x',2),
                               self.W[:,:,i,:].dimshuffle(0,1,'x',2),
                               image_shape=input_shape_row,
                               filter_shape=filter_shape_row,
                               border_mode=self.border_mode,
                               ))

            conved = T.concatenate(conveds,axis=2)



        elif self.border_mode == 'same':
            raise NotImplementedError("Not implemented yet ")
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)


        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x',0,1,'x')


        return self.nonlinearity(activation)
