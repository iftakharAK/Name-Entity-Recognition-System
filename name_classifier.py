import fire


def save_model(model, out_folder: str):
    """
    Serialise the model to an output folder 
    """
    pass


def evaluate_model(model, test_data):
    """
    Evaluate your model against the test data.
    """
    pass


def split_data(data):
    """
    Generate data splits
    """
    pass


def load_data(in_folder: str):
    """
    The in_folder will contain two files:
     - person.ttl
     - name.ttl

    You will need to combine the data to generate the y values (0 or 1),
    and train the model (see readme).
    """
    pass


def train(in_folder: str, out_folder: str) -> None:

    print(in_folder,'  ',out_folder)
    """
    Consume the data from the input folder to generate the model
    and serialise it to the out_folder.
    """
    pass
  

if __name__ == '__main__':
  fire.Fire(train)
