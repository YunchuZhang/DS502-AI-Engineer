import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """
    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(l, if_pool=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?
    #pad= (kernel-stride) /2
    l = mx.sym.Convolution(data=l, kernel=(3, 3), stride =(1, 1),
                           pad=(1, 1), num_filter=20, no_bias=True)
    l = mx.sym.BatchNorm(data = l)
    l = mx.sym.Activation(data = l, act_typr="relu")

    if if_pool:
        l = mx.sym.Pooling( data = l, kernel=(3, 3), stride =(2, 2) ,pool_typr= "max")


    return l
# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?

    l = conv_layer(data, if_pool = False)
    l = conv_layer(l, if_pool = True)
    l = conv_layer(l, if_pool = True)

    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp
if __name__ == " __main__ " :
    cnn = get_conv_sym()
    plot = mx.viz.plot_network(cnn, title="cnn ",
                               save_format= "pdf" , hight_weight=True)
    plot.render("cnn")