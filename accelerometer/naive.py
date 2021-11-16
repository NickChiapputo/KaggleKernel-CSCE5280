import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np

from util import make_confusion_matrix
from util import accumulate_data
from util import aggregate_data
from util import process_data
from util import get_gesture_data
from util import truncate_data
from util import train_test_split
from util import print_results_table
from util import save_results

from models import create_lstm_model, train_model


def main():
    ### Define simulation parameters.
    data_truncate_pad_length = 26
    epochs = 15
    num_iterations = 5
    num_gestures = 20
    num_train_users = 7
    classes = np.arange( num_gestures )

    save_simulation_results = False
    show_confusion_matrix = False


    # Define naive model parameters.
    dropout_rate = 0.0
    lstm_units = 16

    lr = 0.001
    model_optimizer = Adam( learning_rate=lr, decay=1e-6 )

    monitor = 'loss'
    min_delta = 0.001
    patience = 20
    earlystop_callback = EarlyStopping( monitor=monitor, min_delta=min_delta,
                                        verbose=1, patience=patience,
                                        restore_best_weights=True )
    # lstm_callbacks = [earlystop_callback]
    lstm_callbacks = None

    model_params = ( dropout_rate, lstm_units, data_truncate_pad_length, model_optimizer )
    fit_params = ( epochs, lstm_callbacks )


    # Load data and trim/pad to set length.
    train, test = get_data( data_length=data_truncate_pad_length )
    X_test = np.array( [ np.array( test[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T


    # Create data structures to hold confusion matrix and loss/accuracy results
    # for each iteration.
    cf_matrix_true = np.array( [] )
    cf_matrix_pred = np.array( [] )
    scores = []
    user_selections = []


    for i in range( num_iterations ):
        model = create_lstm_model( num_classes=num_gestures,
                                   dropout=dropout_rate,
                                   units=lstm_units,
                                   data_length=data_truncate_pad_length,
                                   optimizer=model_optimizer )

        score, y_test, y_pred,\
        train_val_test_split = train_model( i, train, model,
                                            fit_params, classes=classes,
                                            num_subjects=num_train_users,
                                            verbose=1 )
        model.summary()
        exit(0)

        # Save the train, validation, and test users and the test scores
        # for this iteration.
        user_selections.append( train_val_test_split )
        scores.append( score )

        # Predict on the test dataset and save the
        # submission file for this iteration.
        submission = test.assign( gesture=np.argmax( model.predict( X_test ), axis=1 ) )
        submission.to_csv( f"./submission{i}.csv", index=False,
                           columns=[ 'id', 'gesture' ] )

        # Generate data for confusion matrix.
        cf_matrix_true = np.hstack( ( cf_matrix_true, y_test ) )
        cf_matrix_pred = np.hstack( ( cf_matrix_pred, y_pred ) )

        # Logging output for current iteration.
        print( "test loss, test acc: ", score )


    # Generate the confusion matrix.
    cf_matrix = confusion_matrix( cf_matrix_true, cf_matrix_pred )


    # Print the results for each simulation run in a tabular format.
    print_results_table( scores, user_selections, cf_matrix )


    # Save results.
    if save_simulation_results:
        save_results( scores, user_selections, cf_matrix, epochs,
                      filename=f"results_{epochs}-epochs_"
                               f"{num_iterations}-iterations",
                      loc='./results/', filetype=0 )


    if show_confusion_matrix:
        # Plot the confusion matrix.
        make_confusion_matrix( cf_matrix, categories=classes,
                               figsize=[8,8])


def get_data( data_length ):
    """
    Process and read in the gesture data.

    :param data_length: Desired length to truncate the accelerometer data to. If
    None, do not truncate/pad data.

    :return: Gesture data
    """

    ############################################################################
    #### Below are examples of how to process the data.
    # Accumulate data from original text files into a single (pandas dataframe).
    # Data frame columns are user, gesture, iteration (attempt number), millis,
    # nano, timestamp, accel0, accel1, accel2.
    # Each row is a separate sample from the accelerometer.
    # data = accumulate_data( '../tev-gestures-dataset/',
    #                           target='./gesture_data.csv' )

    # Take the data from accumulate_data and aggregate the iterations so that
    # each row is a single gesture attempt (iteration). Removes the millis,
    # nano, and timestamp.
    # data = aggregate_data( './gesture_data.csv',
    #                           target='./aggregated_gesture_data.csv',
    #                           dir_path=None )

    # After accumulating the data, scale it so that each accelerometer axis has
    # a zero mean and unit variance. This scaling is done per gesture attempt
    # and per axis (you can test a couple of samples to verify that the mean is
    # approximately zero and the variance is approximately 1).
    # data = process_data( './aggregated_gesture_data.csv',
    #                      target='./processed_gesture_data.csv', dir_path=None )
    ############################################################################


    # Load in the pre-processed data.
    train = get_gesture_data( './tev/train.csv' )
    test = get_gesture_data( './tev/test.csv' )

    # Truncate the data as desired.
    train = truncate_data( train, length=data_length )
    test = truncate_data( test, length=data_length )

    return train, test


if __name__ == "__main__":
    main()
