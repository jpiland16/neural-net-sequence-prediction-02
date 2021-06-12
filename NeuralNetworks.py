# PyTorch imports
import torch
from torch import nn

# Personal code
from Util import confirm

# Check to see if we can use the GPU
if torch.cuda.is_available() and confirm("\nGPU found! Use GPU?"):
    device = torch.device('cuda')
    print("\nUsing GPU...")
else:
    device = torch.device('cpu')
    print("\nUsing CPU...")

def get_device():
    return device

device = get_device()

# ------------------------------------------------------------------------------
# NEURAL NETWORK CLASSES -------------------------------------------------------
# ------------------------------------------------------------------------------

class SimpleRNN_01(nn.Module):
    """
    A simple recurrent neural network with 1 hidden layer.
    Based on the tutorial available at 
    https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
    """
    def __init__(self, input_size, output_size, hidden_layer_size, 
            n_layers):
        
        super(SimpleRNN_01, self).__init__()

        # Defining some parameters ------------------------
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers

        #Defining the layers ------------------------------
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_layer_size, 
            n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input: torch.Tensor):
        """
        Return the output(s) from each timestep in the input sequence.
        """
        
        input_size = input.size(0)

        # Initializing hidden state for first input using method defined below
        hidden_state = self.init_hidden(input_size)

        # Passing in the input and hidden state into the model and 
        # obtaining outputs
        output, hidden_state = self.rnn(input, hidden_state)
        
        # Reshaping the outputs such that it can be fit into the 
        # fully connected layer
        output = output.contiguous().view(-1, self.hidden_layer_size)
        output = self.fc(output)
        return output
    
    def init_hidden(self, batch_size):
        """
        Initialize the neural network's hidden layer.
        """
        # This method generates the first hidden state of zeros
        # which we'll use in the forward pass, also sends the tensor 
        # holding the hidden state to the device specified 
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_layer_size)
        hidden = hidden.to(device)
        return hidden
