"""
Tests for ensuring proper functionality of all functions in utils package

Author: mivler
Created: 10/20/2023
Last Modified: 10/20/2023
"""

from . import train_valid_test_split
import pandas as pd
import pytest


def test_train_valid_test_split_single_dataset():
    """
    Tests the train_valid_test_split function on a dummy dataset for a singular array to split    
    """
    # Read in example dataframe (100x3 dataframe)
    df = pd.read_csv("data/testing_data/100x3_df.csv")

    # Run function we are testing
    # Get splits of data as train, valid, and test splits
    train_split, valid_split, test_split = train_valid_test_split(df["col_0"], train_size=.75, valid_size=.15, test_size=.1, random_state=42)

    # Check lengths of split files equals 1, since only 1 dataset was put in to split
    assert len(train_split) == 1
    assert len(valid_split) == 1
    assert len(test_split) == 1

    # Check dataset is properly partitioned
    assert len(train_split[0]) == 75
    assert len(valid_split[0]) == 15
    assert len(test_split[0]) == 10


def test_train_valid_test_split_multi_dataset():
    """
    Tests the train_valid_test_split function on a dummy dataset for 3 arrays to split
    """
    # Read in example dataframe (100x3, dataframe)
    df = pd.read_csv("data/testing_data/100x3_df.csv")

    # Run function we are testing
    # Get splits of data as train, valid, and test splits
    train_split, valid_split, test_split = train_valid_test_split(df["col_0"], df["col_1"], df["col_2"], train_size=.65, valid_size=.20, test_size=.15, random_state=42)

    # Check lengths of split files equals 3
    assert len(train_split) == 3
    assert len(valid_split) == 3
    assert len(test_split) == 3

    # Check correct output for each dataset
    for i in range(3):
        # Check each dataset is properly partitioned
        assert len(train_split[i]) == 65
        assert len(valid_split[i]) == 20
        assert len(test_split[i]) == 15
        # Check values are unchanged and correct after split
        # Check arrays are positionally aligned among all sets
        assert all([elem == i for elem in train_split[i]])
        assert all([elem == i for elem in valid_split[i]])
        assert all([elem == i for elem in test_split[i]])


def test_train_valid_test_split_default():
    """
    Tests the train_valid_test_split function on a dummy dataset for 3 arrays to split
    using default train_size, valid_size, and test_size parameters
    """
    # Read in example dataframe (100x3, dataframe)
    df = pd.read_csv("data/testing_data/100x3_df.csv")

    # Run function we are testing
    # Get splits of data as train, valid, and test splits
    train_split, valid_split, test_split = train_valid_test_split(df["col_0"], df["col_1"], df["col_2"], random_state=42)

    # Check lengths of split files equals 3
    assert len(train_split) == 3
    assert len(valid_split) == 3
    assert len(test_split) == 3

    # Check each dataset is properly partitioned
    for i in range(3):
        assert len(train_split[i]) == 80
        assert len(valid_split[i]) == 10
        assert len(test_split[i]) == 10
        # Check values are unchanged and correct after split
        # Check arrays are positionally aligned among all sets
        assert all([elem == i for elem in train_split[i]])
        assert all([elem == i for elem in valid_split[i]])
        assert all([elem == i for elem in test_split[i]])


def test_train_valid_test_split_error():
    """
    
    """
    # Read in example dataframe (100x3, dataframe)
    df = pd.read_csv("data/testing_data/100x3_df.csv")

    with pytest.raises(AssertionError) as exc_info:
        # Run function we are testing
        # Get splits of data as train, valid, and test splits
        train_split, valid_split, test_split = train_valid_test_split(df["col_0"], df["col_1"], df["col_2"], train_size=.8, valid_size=.20, test_size=.15, random_state=42)

    assert exc_info.type is AssertionError
    assert exc_info.value.args[0] == "Data split proportions do not add up to 1"



