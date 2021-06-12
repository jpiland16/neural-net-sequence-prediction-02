# Ensure the user knows we are starting the program
print("\nImporting necessary packages...")

# Python stardard libraries
import random, pickle

# Third-party libraries
import torch
from torch import nn, Tensor
from tqdm import tqdm

# Personal code
from Sequence import Sequence, generate_sequence
from NeuralNetworks import SimpleRNN_01
from NNParameters import get_parameters
from Util import confirm, one_hot_encode
from NeuralNetworks import get_device

device = get_device()

def get_train_set(parent_seq: Sequence, 
        seq_length: int, num_ex: int) -> 'tuple[Tensor]':
    """
    Generate a pair of training sequence Tensors (X[], y[]) where y is one 
    time-step ahead of X.

    A note on the shape of the input - 3 dimensions:

    For each training example:
        For each timestep in the sequence:
           There should be a value for each input
           (consider this dimension the one-hot encoding)
    """

    train_x = []
    train_y = []
    for _ in range(num_ex):
        rand_start_pos = random.randint(0, len(parent_seq.list) - 1)

        # Notice: the input is one-hot-encoded, but the target is not
        train_x.append(one_hot_encode(
            parent_seq.subseq_from(rand_start_pos, seq_length)))
        train_y.append(
            parent_seq.subseq_from(rand_start_pos + 1, seq_length).list)

    return ( Tensor(train_x), Tensor(train_y) )



def trainSimpleRNN():
    """
    Trains `SimpleRNN_01` from `NeuralNetworks.py` on a 
    newly generated sequence of integers
    """
    # Get necessary parameters
    params = get_parameters("TRAINING", 
        default=confirm("Do you want to use the default parameters?")
    )
    
    # Get a new sequence
    seq = generate_sequence(unique_items=params["NUM_UNIQUE"], 
        length=params["SEQ_LENGTH"])

    # Instantiate the model with hyperparameters
    model = SimpleRNN_01(input_size=seq.unique_items, 
        output_size=seq.unique_items, hidden_layer_size=params["HIDDEN_DIM"], 
        n_layers=params["NUM_LAYERS"])
    
    # Set the model to the device that we defined earlier (default is CPU)
    model.to(device)

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

    # In this case, we are using the same radnomly generated batch 
    # for each training epoch
    training_input, training_target_output = get_train_set(
        seq, params["SUBSEQ_LEN"], params["BATCH_SIZE"])

    # Move to GPU, if available
    training_input = training_input.to(device)
    training_target_output = training_target_output.to(device)

    iter = tqdm(range(params["NUM_EPOCHS"]))
    for _ in iter:
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        output = model(training_input)
        training_target_output = training_target_output.view(-1).long()
        loss = criterion(output, training_target_output)
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly

        iter.set_description(f"loss: {round(loss.item(), 3):5}")

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("\nDone training!\n")
