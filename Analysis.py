# Python standard libraries
import random, pickle, sys

# Third-party libraries
import torch
from torch import nn, Tensor
from tqdm import tqdm

# Personal code
from Sequence import Sequence, get_sequence
from Util import one_hot_encode, confirm
from NNParameters import get_parameters
from NeuralNetworks import get_device

device = get_device()

def get_model() -> nn.Module:
    try:
        pkl = open('model.pkl', "rb")
        model = pickle.load(pkl)
        return model
    except FileNotFoundError:
        print("Could not find model.pkl")
        sys.exit()

def analyze_predictions():

    total_num_predictions = 0
    predictions_correct = 0

    params = get_parameters("TESTING", 
        default=confirm("Do you want to use the default parameters?")
    )

    print("\nLoading model...\n")

    seq = get_sequence()
    model = get_model()
    model = model.to(device)

    iter = range(params["NUM_TESTS"])

    for i in (tqdm(iter) if not params["SHOW_OUTPUT"] else iter):
        random_start_pos = random.randint(0, len(seq.list) - 1)
        input_seq = seq.subseq_from(random_start_pos, params["INPUT_SIZE"])
        target_seq = seq.subseq_from(random_start_pos + 1, params["INPUT_SIZE"])
        output = model_predict(model, input_seq)

        if params["SHOW_OUTPUT"]:
            print(f"Trial {i}\n"
                   " INPUT | TARGET | OUTPUT | ?\n" + 
                   "-------|--------|--------|---")

        for v_in, v_tgt, v_out in zip(input_seq.list, target_seq.list, output):
            if params["SHOW_OUTPUT"]:
                print("{:^7}|{:^8}|{:^8}| {} ".format(v_in, v_tgt, v_out,
                    "Y" if v_out == v_tgt else "-"))
            if v_out == v_tgt:
                predictions_correct += 1
            total_num_predictions += 1

        if params["SHOW_OUTPUT"]:
            print( "-------|--------|--------|---\n")

    print("Predictions correct: " + 
        f"{predictions_correct}/{total_num_predictions} " + 
        f"({round(predictions_correct/total_num_predictions * 100, 2)}%)\n")

def model_predict(model: nn.Module, seq: Sequence):
    # Surround input_seq with square brackets because it is a single batch 
    input_seq = Tensor([one_hot_encode(seq)])
    input_seq = input_seq.to(device)
    output = model(input_seq)

    # Calculate the probabilities at each timestep
    probs_at_each_timestep = \
        [nn.functional.softmax(output[i], dim=0).data for i in range(0,
            len(output) )]

    # Select the index of maximum probability
    return [torch.max(probs, dim=0)[1].item() \
        for probs in probs_at_each_timestep ]