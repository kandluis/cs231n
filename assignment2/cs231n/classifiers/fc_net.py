from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU with
    batch normalization.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-batch norm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    db, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
    dx, dw, db = affine_backward(db, fc_cache)
    return dx, dw, db, dgamma, dbeta


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        #######################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        #######################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        #######################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        hidden, hidden_cache = affine_relu_forward(X, W1, b1)
        W2, b2 = self.params['W2'], self.params['b2']
        scores, cache = affine_forward(hidden, W2, b2)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #######################################################################
        loss, df = softmax_loss(scores, y)
        dhidden, dW2, db2 = affine_backward(df, cache)
        _, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)

        # Bias terms are not regularized.
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        #######################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, self.num_layers + 1):
            self.params['W%s' % i] = np.random.normal(
                0, weight_scale, (dims[i-1], dims[i]))
            self.params['b%s' % i] = np.zeros(dims[i])
        if self.use_batchnorm:
            for i in range(1, self.num_layers):
                self.params['gamma%s' % i] = np.ones(dims[i])
                self.params['beta%s' % i] = np.zeros(dims[i])
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        #######################################################################
        caches = {}
        layer_input = X
        if self.use_batchnorm:
            for i in range(1, self.num_layers):
                layer_input, cache = affine_bn_relu_forward(layer_input,
                    self.params['W%s' % i], self.params['b%s' % i],
                    self.params['gamma%s' % i], self.params['beta%s' % i],
                    self.bn_params[i-1])
                dropout_cache = None
                if self.use_dropout:
                    layer_input, dropout_cache = dropout_forward(layer_input,
                        self.dropout_param)
                caches[i] = (cache, dropout_cache)
        else:
            for i in range(1, self.num_layers):
                layer_input, cache = affine_relu_forward(layer_input,
                    self.params['W%s' % i], self.params['b%s' % i])
                dropout_cache = None
                if self.use_dropout:
                    layer_input, dropout_cache = dropout_forward(layer_input,
                        self.dropout_param)
                caches[i] = (cache, dropout_cache)
        scores, last_cache = affine_forward(layer_input,
                self.params['W%s' % self.num_layers],
                self.params['b%s' % self.num_layers])
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #######################################################################
        loss, df = softmax_loss(scores, y)
        dhidden, dW, db = affine_backward(df, last_cache)
        dgamma, dbeta = None, None
        if self.use_batchnorm:
            for i in reversed(range(1, self.num_layers + 1)):
                dW += self.reg * self.params['W%s' % i]
                grads['W%s' % i] = dW
                grads['b%s' % i] = db
                if dgamma is not None: grads['gamma%s' % i] = dgamma
                if dbeta is not None: grads['beta%s' % i] = dbeta
                if i > 1:
                    cache, dropout_cache = caches[i-1]
                    if self.use_dropout:
                        dhidden = dropout_backward(dhidden, dropout_cache)
                    dhidden, dW, db, dgamma, dbeta = affine_bn_relu_backward(
                        dhidden, cache)
        else:
            for i in reversed(range(1, self.num_layers + 1)):
                dW += self.reg * self.params['W%s' % i]
                grads['W%s' % i] = dW
                grads['b%s' % i] = db
                if i > 1:
                    cache, dropout_cache = caches[i-1]
                    if self.use_dropout:
                        dhidden = dropout_backward(dhidden, dropout_cache)
                    dhidden, dW, db = affine_relu_backward(dhidden, cache)
        # Add regularization.
        for i in range(1, self.num_layers + 1):
            W = self.params['W%s' % i]
            loss += 0.5 * self.reg * np.sum(W * W)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads
