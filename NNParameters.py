ALLOW_ENTER_MEANS_DEFAULT = True

class Parameter():
    """
    Wrapper for a parameter for a neural network.
    """
    def __init__(self, shortname: str, longname: str,
            type: str, value) -> None:
        self.shortname = shortname
        self.longname = longname
        self.type = type
        self.value = value
    
nn_default_params = {
    "TRAINING": [
        Parameter("NUM_LAYERS", "number of layers in RNN", "int", 1),
        Parameter('HIDDEN_DIM', "dimension of hidden layer in RNN", "int", 20),
        Parameter('SUBSEQ_LEN', "subsequence length for training", "int", 10),
        Parameter('BATCH_SIZE', "mini-batch size for training", "int", 300),
        Parameter('NUM_EPOCHS', "number of training epochs", "int", 500),
        Parameter('LEARNING_RATE', "learning rate", "float", 0.05),
        Parameter('NUM_UNIQUE', "number of unique items in sequence", "int", 10),
        Parameter('SEQ_LENGTH', "length of the sequence to repeat", "int", 100)
    ],
    "TESTING": [
        Parameter("NUM_TESTS", "number of tests to run", "int", 1000),
        Parameter("INPUT_SIZE", "length of the subsequence for input", "int", 40),
        Parameter("SHOW_OUTPUT", "show output tables", "int", 0)
    ]
}

def get_parameters(group_id: str, default=True) -> dict:
    """
    Get the parameters governing the neural network, or allow the 
    user to set them if `default` == `False`.
    """
    param_set = {}
    if not default:
        print("Please enter parameters below " +
            "(default values given in parentheses)\n")

        if ALLOW_ENTER_MEANS_DEFAULT:
            print("Press ENTER to accept default for any parameter.\n")

    for parameter in nn_default_params[group_id]:
        if default:
            # Use the default value of the parameter as defined above
            param_set[parameter.shortname] = parameter.value
        else:
            # Ask the user to enter the parameters
            input_accepted = False
            while not input_accepted:
                try:
                    # Ask for input
                    val = input(f"{parameter.longname} " + 
                        f"({parameter.shortname} = {parameter.value}) > ")
                    # Ensure their entry is of the correct type
                    if parameter.type == 'int':
                        val = int(val)
                    elif parameter.type == 'float':
                        val = float(val)
                    param_set[parameter.shortname] = val
                    input_accepted = True
                except KeyboardInterrupt:
                    raise
                except:
                    if ALLOW_ENTER_MEANS_DEFAULT and val == "":
                            param_set[parameter.shortname] = parameter.value
                            input_accepted = True
                    else:
                        print("Invalid entry!")
    return param_set