import pandas as pd
import numpy as np


def add_row_to_dataframe( row, user_iterations, df, df_len,
                          length_cutoff, length_of_gesture ):
    num_attempts = 0

    # Since the data was not recorded properly, some users do not have
    # timestamps that reset after the (approximately) three second
    # window. Instead, we have to arbitrarily divide their recordings
    # into a number of gestures. Sampling rate is about 100 Hz, so we
    # divide the measurements into sets of 300. Any remaining data is
    # discarded.
    if len( row[ 3 ] ) > length_cutoff:
        num_gestures = len( row[ 3 ] ) // length_of_gesture
        for gesture_idx in range(num_gestures):
            curr_row = [
                            row[ 0 ], row[ 1 ],
                            user_iterations[ row[ 0 ] ],
                            row[ 3 ][ length_of_gesture * gesture_idx :
                                      length_of_gesture * ( gesture_idx + 1 ) ],
                            row[ 4 ][ length_of_gesture * gesture_idx :
                                      length_of_gesture * ( gesture_idx + 1 ) ],
                            row[ 5 ][ length_of_gesture * gesture_idx :
                                      length_of_gesture * ( gesture_idx + 1 ) ],
                            row[ 6 ][ length_of_gesture * gesture_idx :
                                      length_of_gesture * ( gesture_idx + 1 ) ]
                       ]
            df.loc[ df_len ] = curr_row

            df_len += 1
            user_iterations[ row[ 0 ] ] += 1

            num_attempts += 1

            print( f"Attempt {df_len:3d}: ({row[ 0 ]}, {row[ 1 ]}, {length_of_gesture})" )
    else:
        curr_row = [ row[ 0 ], row[ 1 ], user_iterations[ row[ 0 ] ],
                     row[ 3 ], row[ 4 ], row[ 5 ], row[ 6 ] ]
        df.loc[ df_len ] = curr_row

        df_len += 1
        user_iterations[ row[ 0 ] ] += 1

        num_attempts += 1

        print( f"Attempt {df_len:3d}: ({row[ 0 ]}, {row[ 1 ]}, {len( row[ 3 ] )})" )

    return num_attempts

def get_p1_data( path ):
    # Create an empty DataFrame that will hold the aggregated P1 data.
    p1_agg = pd.DataFrame( columns=['user', 'gesture', 'iteration',
                                    'x', 'y', 'z', 'absolute'] )

    # Get the data from the CSV file.
    p1_data = pd.read_csv( path )
    unique_users = list( p1_data.Part_Name.unique() )
    unique_gestures = list( p1_data.Motion.unique() )
    print( unique_users )
    print( unique_gestures )
    # exit(0)

    user_iterations = np.zeros( len( unique_users ) )

    # These are used to track the data from the current gesture attempt
    # while we go row-by-row over the data set.
    curr_x_data = []
    curr_y_data = []
    curr_z_data = []
    curr_abs_data = []
    curr_user = unique_users.index( p1_data.iloc[ 0 ][ 'Part_Name' ] )
    curr_gesture = unique_gestures.index( p1_data.iloc[ 0 ][ 'Motion' ] )

    # P1 data is appropriately sliced every 3 seconds, but only for the first
    # user and the first gesture. The remaining users and gestures record for
    # varying lengths of time. For these situations, we simply slice the
    # recordings into 300 samples each (3 seconds at 100 Hz) and discard the
    # remaining readings.
    # Read line by line until the timestamp of the next data is less than the
    # timestamp of the previous line. This is the demarcation between gesture
    # attempts. At this point, save previous data as a single gesture and push
    # it to the final DataFrame.
    total_num_attempts = 0
    total_measurements = 0
    last_timestamp = 0
    for idx, row in p1_data.iterrows():
        # Current row starts new attempt.
        if row[ 'Time (s)' ] < last_timestamp and idx > 0:
            curr_row = [ curr_user, curr_gesture, user_iterations[ curr_user ],
                         curr_x_data, curr_y_data, curr_z_data, curr_abs_data ]
            new_attempts = add_row_to_dataframe( curr_row, user_iterations, p1_agg,
                                                 total_num_attempts, 350, 300 )

            user_iterations[ curr_user ] += new_attempts
            total_num_attempts += new_attempts

            curr_user = unique_users.index( row[ 'Part_Name' ] )
            curr_gesture = unique_gestures.index( row[ 'Motion' ] )
            curr_x_data = []
            curr_y_data = []
            curr_z_data = []
            curr_abs_data = []
        elif unique_users.index( row[ 'Part_Name' ] ) != curr_user:
            print( f"ERROR: Current user does not match expected. "
                   f"{unique_users.index(row['Part_Name'])} != {curr_user}.")
            exit(0)
        elif unique_gestures.index( row[ 'Motion' ] ) != curr_gesture:
            print( f"ERROR: Current gesture does not match expected. "
                   f"{unique_gestures.index(row['Motion'])} != {curr_gesture}.")
            exit(0)

        curr_x_data.append( row[ 'Acceleration x (m/s^2)' ] )
        curr_y_data.append( row[ 'Acceleration y (m/s^2)' ] )
        curr_z_data.append( row[ 'Acceleration z (m/s^2)' ] )
        curr_abs_data.append( row[ 'Absolute acceleration (m/s^2)' ] )

        total_measurements += 1

        last_timestamp = row[ 'Time (s)' ]

    # Add the last gesture attempt(s).
    curr_row = [ curr_user, curr_gesture, user_iterations[ curr_user ],
                 curr_x_data, curr_y_data, curr_z_data, curr_abs_data ]
    new_attempts = add_row_to_dataframe( curr_row, user_iterations, p1_agg,
                                         total_num_attempts, 350, 300 )

    user_iterations[ curr_user ] += new_attempts
    total_num_attempts += new_attempts

    return p1_agg


def get_p2_data( path ):
    # Create an empty DataFrame that will hold the aggregated P1 data.
    p2_agg = pd.DataFrame( columns=['user', 'gesture', 'iteration',
                                    'x', 'y', 'z', 'absolute'] )

    # Get the data from the CSV file.
    p2_data = pd.read_csv( path )
    unique_users = list( p2_data.Part_Name.unique() )
    unique_gestures = list( p2_data.Motion.unique() )
    print( unique_users )
    print( unique_gestures )
    # exit(0)

    user_iterations = np.zeros( len( unique_users ) )

    # These are used to track the data from the current gesture attempt
    # while we go row-by-row over the data set.
    curr_x_data = []
    curr_y_data = []
    curr_z_data = []
    curr_abs_data = []
    curr_user = unique_users.index( p2_data.iloc[ 0 ][ 'Part_Name' ] )
    curr_gesture = unique_gestures.index( p2_data.iloc[ 0 ][ 'Motion' ] )
    gesture_start_time = p2_data.iloc[ 0 ][ 'Time (s)' ]

    # P2 data is not sliced except for the switch from one hand to the next.
    # Additionally, the sampling rate is supposedly
    # Read line by line until the timestamp of the next data is less than the
    # timestamp of the previous line. This is the demarcation between gesture
    # attempts. At this point, save previous data as a single gesture and push
    # it to the final DataFrame.
    total_num_attempts = 0
    total_measurements = 0
    last_timestamp = 0
    for idx, row in p2_data.iterrows():
        # if curr_user == 0 and curr_gesture == 1:
        #     exit(0)

        # Current row starts new attempt.
        # There is one case for user 2 and gesture 0 in P2 where the timestamp
        # goes backwards by about 0.01 seconds, so we add a small offset of 0.02
        # to account for this.
        if row[ 'Time (s)' ] < last_timestamp and \
           row[ 'Time (s)' ] - last_timestamp < -0.02 and \
           idx > 0:
            curr_row = [ curr_user, curr_gesture, user_iterations[ curr_user ],
                         curr_x_data, curr_y_data, curr_z_data, curr_abs_data ]
            new_attempts = add_row_to_dataframe( curr_row, user_iterations, p2_agg,
                                                 total_num_attempts, 10000, 1500 )
            print( f"    {gesture_start_time}-{p2_data.iloc[ total_measurements - 1 ][ 'Time (s)' ]} "
                   f"({p2_data.iloc[ total_measurements - 1 ][ 'Time (s)' ] - gesture_start_time})" )

            gesture_start_time = row[ 'Time (s)' ]

            user_iterations[ curr_user ] += new_attempts
            total_num_attempts += new_attempts

            curr_user = unique_users.index( row[ 'Part_Name' ] )
            curr_gesture = unique_gestures.index( row[ 'Motion' ] )
            curr_x_data = []
            curr_y_data = []
            curr_z_data = []
            curr_abs_data = []
        elif unique_users.index( row[ 'Part_Name' ] ) != curr_user:
            print( f"ERROR: Current user does not match expected. "
                   f"{unique_users.index(row['Part_Name'])} != {curr_user}.")
            exit(0)
        elif unique_gestures.index( row[ 'Motion' ] ) != curr_gesture:
            print( f"ERROR: Current gesture does not match expected. "
                   f"{unique_gestures.index(row['Motion'])} != {curr_gesture}.")
            exit(0)

        curr_x_data.append( row[ 'Acceleration x (m/s^2)' ] )
        curr_y_data.append( row[ 'Acceleration y (m/s^2)' ] )
        curr_z_data.append( row[ 'Acceleration z (m/s^2)' ] )
        curr_abs_data.append( row[ 'Absolute acceleration (m/s^2)' ] )

        total_measurements += 1

        last_timestamp = row[ 'Time (s)' ]

    # Add the last gesture attempt(s).
    curr_row = [ curr_user, curr_gesture, user_iterations[ curr_user ],
                 curr_x_data, curr_y_data, curr_z_data, curr_abs_data ]
    new_attempts = add_row_to_dataframe( curr_row, user_iterations, p2_agg,
                                         total_num_attempts, 10000, 1500 )

    user_iterations[ curr_user ] += new_attempts
    total_num_attempts += new_attempts

    return p2_agg

def main():
    # Read in the data from P1
    # p1_data = get_p1_data( './p1/MasterList.csv' )
    # p1_data.to_csv( './p1/aggregated.csv', index=False )

    # Read in the data from P2
    p2_data = get_p2_data( './p2/MasterList.csv' )
    # p2_data = pd.to_csv( './p2/aggregated.csv', index=False )

    # Combine it
    return

if __name__ == "__main__":
    main()