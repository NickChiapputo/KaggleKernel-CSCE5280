import pandas as pd
import numpy as np
from numpy.random import randint
from ast import literal_eval


def get_processed_data():
    converters = {
        header: literal_eval for header in [ 'x', 'y', 'z' ]
    }
    return pd.read_csv( './processed_gesture_data.csv', converters=converters )


def insert_id_column( path='./processed_gesture_data.csv', data=None, target=None ):
    if path is None and data is None:
        raise ValueError( "Either path or data must not be None." )

    if path is not None:
        converters = {
            header: literal_eval for header in [ 'x', 'y', 'z' ]
        }
        data = pd.read_csv( path, converters=converters )

    data.insert( 0, 'id', range( len( data ) ) )

    if target is not None:
        data.to_csv( target, index=False )
    return data


def split_data( data, test_users=None ):
    # If we aren't given a list of test users, randomly
    # select a single user over the range [0, 7).
    if test_users is None:
        test_users = [ randint( 8 ) ]

    sel_column = 'user'

    test = data[ data[ sel_column ].isin( test_users ) ]
    train = data[ ~data[ sel_column ].isin( test_users ) ]

    # Randomly shuffle train and test.
    train = train.sample( frac=1, random_state=42 )
    test = test.sample( frac=1, random_state=42 )

    return train, test


def get_submission_df( test ):
    # Remove all but the id and gesture.
    return test[ [ 'id', 'gesture' ] ]


def main():
    data = get_processed_data()
    train, test = split_data( data, test_users=[7] )
    train = insert_id_column( path=None, data=train, target=None )
    test = insert_id_column( path=None, data=test, target=None )

    # Save data to csv format. For test data, don't save gestures.
    train.to_csv( './train.csv', index=False, columns=['id', 'user', 'gesture', 'x', 'y', 'z'] )
    test.to_csv( './test.csv', index=False, columns=['id', 'user', 'x', 'y', 'z'] )

    solution = get_submission_df( test )
    solution.to_csv( './solution.csv', index=False )


if __name__ == "__main__":
    main()