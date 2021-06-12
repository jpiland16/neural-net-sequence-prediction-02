# Sequence prediction with neural networks

This repository aims to outline some of the basics of using an LSTM to
echo a random sequence of numbers that repeats.

## Project strucutre

 - **Analysis.py**
   - `get_model`: retrieves model from disk
   - `model_predict`: returns output(s) of neural network given an input 
      sequence
   - `analyze_predictions`: loads sequence and model and shows the accuracy
      and results of the model
 - **Main.py**
   - Holds the main user interface 
     __*(Run `python Main.py` to use the program)*__
 - **NeuralNetworks.py**
   - Contains neural network classes as well as a GPU check
   - Current neural networks:
     - `SimpleRNN_01`: a simple RNN with 1 hidden layer and 1 output layer
 - **NNParameters.py**
   - Stores default parameters for model training and testing
   - Also contains a function to ask the user to provide values for each
     parameter
 - **Sequence.py**
   - Contains wrapper class `Sequence` for use in training and testing
   - Contains functions for generating a random sequence and retrieving it
     from disk
 - **Training.py**
   - Contains functions to obtain the training set
     as well as to train `SimpleRNN_01`
 - **Util.py**
   - Contains sequence operations and a user-confirmation function

## Notes & changelog

### 6/12/2021 - [`initial commit`](https://github.com/jpiland16/neural-net-sequence-prediction-02/tree/c026506cf8ea36945ed4f8db750cc78677f9f543)

Implemented a basic RNN, loosely following 
[this tutorial at FloydHub](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/).

What I learned:
 - When training a model using cross-entropy loss, the input-sequence 
   parameter has a different shape than the target-sequence parameter. 
   Specifically, the training input should consist of the following:

     - For each **batch**<sup>[1](#batchnote)</sup> in the set of training sequences:

       - For each timestep in the sequence

         - There should be an array of values - one for each output of the
           network (in this case, our one-hot encoding)

   So the input has shape (# of sequences, # of timesteps, # of outputs )
   and *# of outputs* is the number of unique items in the sequence. *(# of timesteps = sequence length)*

 - In contrast, the target sequence should have the following structure:

    - For each **batch**<sup>[1](#batchnote)</sup> in the set of output sequences:

      - For each timestep in the sequence
        
        - There should be a single value corresponding to the correct
          index that should be active

   So the target has shape (# of sequences, # of timesteps).

 - The output of the neural network is of shape (# of timesteps, # of 
   outputs.)

Outstanding questions:
 - Is the neural network really learning to recognize the sequence or is it 
   just memorizing it (or are these the same)?
 - What would happen if the input sequence contained random "jumps" to other
   parts of the sequence? Could this improve/harm the neural network?
 - Could the network benefit from LSTM or other types of layers?
 - Could the network benefit from dropout?
 - Could training be improved by using different batches (in the sense
   of *groups of sequences*) in each epoch and ensuring the batches 
   were unique?

TODO:
 - Attempt to solve some of these questions
 - Modify `Training.py` to be able to train any model

---

### Footnotes

<a name="batchnote">1</a>: I don't like this use of the term *batch* - because in this case it seems to be actually referring to a single input sequence, rather than a group of input sequences.