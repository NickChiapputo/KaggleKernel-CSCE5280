import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

from util import get_gesture_data
from util import truncate_data
from util import make_confusion_matrix
from util import print_results_table
from util import save_results

from models import create_lstm_model
from models import train_model


def main():
    # Define simulations parameters.
    data_truncate_pad_length = 300
    epochs = 50
    num_iterations = 2
    num_gestures = 6
    classes = np.arange( num_gestures )

    save_simulation_results = False
    show_confusion_matrix = True


    # Define model parameters.
    dropout_rate = 0.6
    lstm_units = 256

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


    # Load the data and trim/pad to set length.
    data = get_data( data_length=300 )


    # Create data structures to hold confusion matrix and loss/accuracy results
    # for each iteration.
    cf_matrix_true = np.array( [] )
    cf_matrix_pred = np.array( [] )
    scores = []
    user_selections = []


    # Run model training simulations for desired number of iterations.
    for i in range( num_iterations ):
        model = create_lstm_model( num_classes=num_gestures,
                                   dropout=dropout_rate,
                                   units=lstm_units,
                                   data_length=data_truncate_pad_length,
                                   optimizer=model_optimizer )

        score, y_test, y_pred,\
        train_val_test_splits = train_model( i, data, model,
                                             fit_params, classes=classes,
                                             num_subjects=14,
                                             verbose=1 )

        user_selections.append( train_val_test_splits )

        scores.append( score )

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
        make_confusion_matrix( cf_matrix,
                               categories=classes,
                               figsize=[8,8])


def get_data( data_length ):
    """
    Process and read in the gesture data.

    :param data_length: Desired length to truncate the accelerometer data to. If
    None, do not truncate/pad data.

    :return: Gesture data
    """

    # Load in the pre-processed data.
    data = get_gesture_data( './p1/processed_gesture_data.csv' )

    # Truncate the data as desired. Comment out to test non-truncated data.
    # Make sure your model can handle variable length data!
    data = truncate_data( data, length=data_length )

    return data

if __name__ == "__main__":
    main()