# Python libraries
from sys import exit

# Personal code
from Training import trainSimpleRNN
from Analysis import analyze_predictions
from Util import show_seq_str

class Thing():
    """
    Wrapper for a function that provides human-readable 
    command-line summaries.
    """
    def __init__(self, function: 'function', description: str) -> None:
        self.function = function
        self.description = description

things_to_do = [
    Thing(trainSimpleRNN, "Generate a new sequence and train a simple RNN"),
    Thing(analyze_predictions, "Analyze the predictions of the model"),
    Thing(show_seq_str, "Show the saved sequence"),
    Thing(exit, "Exit")
]

def do_something(msg: str):
    """
    Ask the user what they would like to do.
    """
    print(f"\n{msg}\n")
    for index, thing in enumerate(things_to_do):
        print(f" {index} - {thing.description}")
    print()
    while True:
        try:
            input_value = int(input("Enter an option " + 
                f"(0-{len(things_to_do) - 1}) > "))
            if (input_value >= 0) and input_value < len(things_to_do):
                break
        except KeyboardInterrupt:
            raise
        except:
            print("Invalid entry!")
            pass
            
    print()
    things_to_do[input_value].function()

def main():
    while True:
        do_something("What would you like to do? (CTRL-C to exit)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()