from sklearn.model_selection import train_test_split


def train_valid_test_split(*datasets, train_size=.8, valid_size=.1, test_size=.1, random_state=42):
    """
    Splits a set of data into training, validation, and test sets

    Parameters
    ----------
    *datasets : (sequence of indexables with same length / shape[0]) Set of 
    train_size : (float) Proportion of datset for train set
    valid_size : (float) Proportion of dataset for valid set
    test_size : (float) Proportion of dataset for test set
    random_state : (int) Seed for random state in splits

    Return
    ------
    (tuple of lists) [0] list of training splits, [1] list of validation splits, [2] list of test splits
    """
    
    # Check proprotions of datasets add up to 1
    assert abs(1 - (train_size + valid_size + test_size)) < 1e-8, "Data split proportions do not add up to 1"

    # Calculate reweighted validation proportion
    valid_size = valid_size / (train_size + valid_size)

    # Split dataset into test set and all other examples
    output_test_split = train_test_split(*datasets, test_size=test_size, random_state=random_state)
    # Unpack outputs into final test split and the rest of the data
    outputs_test = output_test_split[1::2]
    outputs_rest = output_test_split[0::2]

    # Split remaining examples into train set and validation set
    output_valid_split = train_test_split(*outputs_rest, test_size=valid_size, random_state=random_state//2)
    # Unpack otuputs into final validation split and final training split
    outputs_valid = output_valid_split[1::2]
    outputs_train = output_valid_split[0::2]

    return outputs_train, outputs_valid, outputs_test
