#Author: Tomojit Ghosh
# This module will create a one-to-one connected layer without any bias term.
# This layer will be placed after the input layer. Each node of this layer is
# connected via a weight to a node in the input layer. Applying L1 penalty on
# the weights will drive most of the connection to zero and the coresponding
# input node can be ignored. This way this module can be used as a feature detector.
# The code of this module is inspired from https://github.com/uchida-takumi/CustomizedLinear.

import pdb
import math
import torch
import torch.nn as nn

#################################
# Define custome autograd function for masked connection.

class OneToOneLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    
    def forward(ctx, input, weight):
        #pdb.set_trace()
        output = input*weight.unsqueeze(0).expand_as(input)
        ctx.save_for_backward(input, weight)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #pdb.set_trace()
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            #pdb.set_trace()            
            grad_weight = (grad_output*input).sum(0).squeeze(0)

        return grad_input, grad_weight


class OneToOneLinear(nn.Module):
    def __init__(self, nFeatures):
        """
        extended torch.nn module which mask connection.
        This layer is connected with the input layer by a one-to-one connection.
        Total number of nodes in this layer will be the same as in the input layer.
        Argumens
        ------------------
        nFeatures: No. of features in the input data.
        
        """
        super(OneToOneLinear, self).__init__()
        self.input_features = nFeatures
        
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        
        # initialize the weight to 1
        #pdb.set_trace()
        self.weight = nn.Parameter(torch.Tensor(torch.ones(self.input_features)))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return OneToOneLinearFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
