__author__ = 'Alexandros Armaos  (alexandros@tartaglialab.com )'

import theano
import theano.tensor as T
import numpy
import lasagne
import argparse

import DCNN
import dataUtils
import networks
import utils
import IPython


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

parser = argparse.ArgumentParser(description='Train a DCNN on the binary Stanford Sentiment dataset as specified in the Kalchbrenner \'14 paper. All the default values are taken from the paper or the Matlab code.')
# training settings
parser.add_argument("--learning_rate",type=float, default=0.1, help='Learning rate')
parser.add_argument("--n_epochs",type=int,default=10,help="Number of epochs")
parser.add_argument("--valid_freq",type=int,default=10,help="Number of batches processed until we validate.")
parser.add_argument("--adagrad_reset",type=int,default=5,help="Resets the adagrad cumulative gradient after x epochs. If the value is 0, no reset will be executed.")
parser.add_argument("--nlayers",type=int,default=2,help="Number of convolution layers")
# input output
parser.add_argument("--vocab_size",type=int, default=100, help='Vocabulary size')
parser.add_argument("--output_classes",type=int, default=2, help='Number of output classes')
parser.add_argument("--batch_size",type=int, default=4, help='Batch size')
# network paras
parser.add_argument("--filter_size_conv_layers", nargs="+", type=int, default=[7,5],help="List of sizes of filters at layer 1 and 2, default=[8,5]")
parser.add_argument("--nr_of_filters_conv_layers", nargs="+", type=int, default=[7,15],help="List of number of filters at layer 1 and 2, default=[20,14]")
parser.add_argument("--activations",nargs='+', type=str,default=["tanh","tanh"],help="List of activation functions behind first and second conv layers, default [tanh, tanh]. Possible values are \"linear\", \"tanh\", \"rectify\" and \"sigmoid\". ")
parser.add_argument("--L2",nargs='+',type=float,default=[0.00003/2,0.000003/2,0.0001/2],help="Fine-grained L2 regularization. X values are needed for X layers, namely for the  X-1 conv layers and a final/output dense layer.")
parser.add_argument("--ktop",type=int,default=4,help="K value of top pooling layer DCNN")
parser.add_argument("--dropout_value", type=float,default=0.5,help="Dropout value after penultimate layer")
parser.add_argument("--channels_size", type=int,default=20,help="Number of input channels")
# cost functions
parser.add_argument("--objective", type=str,default='binary',help="Objective binary/categorical crossentropy")


args = parser.parse_args()
hyperparas = vars(args)
print("Hyperparameters: "+str(hyperparas))

if len(hyperparas['filter_size_conv_layers'])!= hyperparas['nlayers'] or len(hyperparas['nr_of_filters_conv_layers'])!=hyperparas['nlayers'] or len(hyperparas['activations'])!=hyperparas['nlayers'] or len(hyperparas["L2"])!=hyperparas['nlayers']+1 :
    raise Exception('Check if the input --filter_size_conv_layers, --nr_of_filters_conv_layers and --activations are lists of size 2, and the --L2 field needs a value list of 4 values.')
if hyperparas['nlayers']<=0:
    raise Exception('Check number of convolution layers!')
title="LR"+str(hyperparas["learning_rate"])+"_NL"+str(hyperparas["nlayers"])+"_L2"+str(hyperparas["L2"])+"_k_"+str(hyperparas["ktop"])+str(hyperparas["objective"])


#######################
# LOAD  TRAINING DATA #
#######################
print('Loading the training data')
training_data_path="data/protein_fragments/"
train_x_indexes, train_y, train_lengths = dataUtils.read_data_1d(training_data_path+"train_x.txt",training_data_path+"train_y.txt")
test_x_indexes, test_y, test_lengths = dataUtils.read_data_1d(training_data_path+"test_x.txt",training_data_path+"test_y.txt")
dev_x_indexes, dev_y, dev_lengths = dataUtils.read_data_1d(training_data_path+"valid_x.txt",training_data_path+"valid_y.txt")

shape_=train_x_indexes.shape
train_x_indexes=train_x_indexes.reshape(shape_[0],shape_[1],1,shape_[2])

shape_=test_x_indexes.shape
test_x_indexes=test_x_indexes.reshape(shape_[0],shape_[1],1,shape_[2])

shape_=dev_x_indexes.shape
dev_x_indexes=dev_x_indexes.reshape(shape_[0],shape_[1],1,shape_[2])
#channels_size=train_x_indexes.shape[1]
n_train_batches = len(train_lengths) / hyperparas['batch_size']


#dev data
# to be able to do a correct evaluation, we pad a number of rows to get a multiple of the batch size
dev_x_indexes_extended = dataUtils.pad_to_batch_size(dev_x_indexes,hyperparas['batch_size'])
dev_y_extended = dataUtils.pad_to_batch_size(dev_y,hyperparas['batch_size'])
n_dev_batches = dev_x_indexes_extended.shape[0] / hyperparas['batch_size']
n_dev_samples = len(dev_y)
dataUtils.extend_lenghts(dev_lengths,hyperparas['batch_size'])

# test data
test_x_indexes_extended = dataUtils.pad_to_batch_size(test_x_indexes,hyperparas['batch_size'])
test_y_extended = dataUtils.pad_to_batch_size(test_y,hyperparas['batch_size'])
n_test_batches = test_x_indexes_extended.shape[0] / hyperparas['batch_size']
n_test_samples = len(test_y)
dataUtils.extend_lenghts(test_lengths,hyperparas['batch_size'])

######################
# BUILD ACTUAL MODEL #
######################
print('Building the model')

# allocate symbolic variables for the data
X_batch = T.dtensor4('x')
y_batch = T.lvector('y')


rng = numpy.random.RandomState(23455)
# define/load the network
output_layer = networks.build1DDCNN_dynamic(nlayers=hyperparas['nlayers'],batch_size=hyperparas['batch_size'],channels_size=hyperparas['channels_size'],vocab_size=hyperparas['vocab_size'],filter_sizes=hyperparas['filter_size_conv_layers'],nr_of_filters=hyperparas['nr_of_filters_conv_layers'],activations=hyperparas['activations'],ktop=hyperparas['ktop'],dropout=hyperparas["dropout_value"],output_classes=hyperparas['output_classes'],padding='last')

l2_layers = []
for layer in lasagne.layers.get_all_layers(output_layer):
    if isinstance(layer,(DCNN.convolutions.Conv1DLayer,lasagne.layers.DenseLayer)):
        l2_layers.append(layer)

if objective=="categorical":
    s1=lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch),y_batch),mode='mean')
if objective=="binary":
    s1=lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(lasagne.layers.get_output(output_layer,X_batch),y_batch),mode='mean')
s2=lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers,hyperparas["L2"])),lasagne.regularization.l2)
loss_train=s1+s2
#loss_train = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch),y_batch),mode='mean')+lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers,hyperparas["L2"])),lasagne.regularization.l2)

# validating/testing
loss_eval = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch,deterministic=True),y_batch)
pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True),axis=1)
correct_predictions = T.eq(pred, y_batch)

# In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
all_params = lasagne.layers.get_all_params(output_layer)
updates, accumulated_grads = utils.adagrad(loss_train, all_params, hyperparas['learning_rate'])
#updates = lasagne.updates.adagrad(loss_train, all_params, hyperparas['learning_rate'])


train_model = theano.function(inputs=[X_batch,y_batch], outputs=loss_train,updates=updates)

valid_model = theano.function(inputs=[X_batch,y_batch], outputs=correct_predictions)

test_model = theano.function(inputs=[X_batch,y_batch], outputs=correct_predictions)



###############
# TRAIN MODEL #
###############
print('Started training')
print('Because of the default high validation frequency, only improvements are printed.')

best_validation_accuracy = 0
epoch = 0
batch_size = hyperparas["batch_size"]
train_costs=[]
validation_accuraces=[]
testing_accuracies=[]
while (epoch < hyperparas['n_epochs']):
    epoch = epoch + 1
    permutation = numpy.random.permutation(n_train_batches)
    batch_counter = 0
    train_loss=0
    for minibatch_index in permutation:
        x_input = train_x_indexes[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,0:train_lengths[(minibatch_index+1)*batch_size-1]]
        y_input = train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
        #print "train", x_input.shape , y_input.shape
        train_loss+=train_model(x_input,y_input)
        train_costs.append(train_loss)

        if batch_counter>0 and batch_counter % hyperparas["valid_freq"] == 0:
            accuracy_valid=[]
            for minibatch_dev_index in range(n_dev_batches):
                x_input = dev_x_indexes_extended[minibatch_dev_index*batch_size:(minibatch_dev_index+1)*batch_size,:,:,0:dev_lengths[(minibatch_dev_index+1)*batch_size-1]]
                y_input = dev_y_extended[minibatch_dev_index*batch_size:(minibatch_dev_index+1)*batch_size]
                #print "dev", x_input.shape , y_input.shape
                accuracy_valid.append(valid_model(x_input,y_input))

            #dirty code to correctly asses validation accuracy, last results in the array are predictions for the padding rows and can be dumped afterwards
            this_validation_accuracy = numpy.concatenate(accuracy_valid)[0:n_dev_samples].sum()/float(n_dev_samples)
            validation_accuraces.append(this_validation_accuracy)
            if this_validation_accuracy > best_validation_accuracy:
                print("Train loss, "+str( (train_loss/hyperparas["valid_freq"]))+", validation accuracy: "+str(this_validation_accuracy*100)+"%")
                best_validation_accuracy = this_validation_accuracy

                # test it
                accuracy_test= []
                for minibatch_test_index in range(n_test_batches):
                    x_input = test_x_indexes_extended[minibatch_test_index*batch_size:(minibatch_test_index+1)*batch_size,:,:,0:test_lengths[(minibatch_test_index+1)*batch_size-1]]
                    y_input = test_y_extended[minibatch_test_index*batch_size:(minibatch_test_index+1)*batch_size]
                    #print "test", x_input.shape , y_input.shape
                    accuracy_test.append(test_model(x_input,y_input))
                this_test_accuracy = numpy.concatenate(accuracy_test)[0:n_test_samples].sum()/float(n_test_samples)
                testing_accuraces_accuraces.append(this_test_accuracy)

                print("Test accuracy: "+str(this_test_accuracy*100)+"%")

            train_loss=0
        batch_counter+=1

        dataUtils.check_plots(title,train_costs,validation_accuraces, testing_accuraces_accuraces)
    if hyperparas["adagrad_reset"] > 0:
        if epoch % hyperparas["adagrad_reset"] == 0:
            utils.reset_grads(accumulated_grads)

    print("Epoch "+str(epoch)+" finished.")
