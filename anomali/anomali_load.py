import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import snowflake.connector as sf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from flask import Flask

# Define the S3 bucket and Snowflake parameters
snowflake_account = 'at67872.ap-south-1.aws'
snowflake_user = 'FYP4'
snowflake_password = '!Q2w3e4r5t6y'
warehouse='COMPUTE_WH'
snowflake_database = 'COAPDATA'
snowflake_schema = 'METER_SCHEMA'

app = Flask(__name__)



@app.route('/detect', methods=['POST'])
def detect():
    # Connect to Snowflake
    conn = sf.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    warehouse=warehouse,
    database=snowflake_database,
    schema=snowflake_schema
    )

    # Get the list of water meter views
    cursor = conn.cursor()
    cursor.execute(f"SHOW VIEWS IN SCHEMA {snowflake_database}.{snowflake_schema}")
    views = [row[1] for row in cursor.fetchall()]
    cursor.close()

    # Dictionary to store the dataframes for each water meter view
    view_dataframes = {}
    dataframes = []

    # Iterate over each water meter view
    for view in views:
        # Retrieve the data from the current water meter view
        query = f"SELECT Date, total FROM {snowflake_database}.{snowflake_schema}.{view} ORDER BY Date DESC"
        df = pd.read_sql(query, conn)
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        dataframes.append(df)

    # # # Scale the data using standardization
    # scaler = StandardScaler()
    # # Convert the 'total' column to a numpy array
    # data = df['TOTAL'].values 
    # scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # # # Store the dataframe in the dictionary
    # view_dataframes[view] = scaled_data


    # Scale each dataframe using StandardScaler
    scaler = StandardScaler()
    scaled_dataframes = []

    for df in dataframes:
        df = df['TOTAL'].values 
        scaled_data = scaler.fit_transform(df.reshape(-1, 1))
        # scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        scaled_dataframes.append(scaled_data)

    import tensorflow as tf
    from tensorflow.keras import layers

    # Define the input layer
    input_layer = Input(shape=(1,))

    # Define the encoder layers
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)

    # Define the decoder layers
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(1, activation='linear')(decoded)

    # Create the autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')


    # List to store the trained autoencoders
    autoencoders = []

    # Define the autoencoder model
    input_dim = scaled_dataframes[0].shape[1]
    encoding_dim = 64

    for scaled_df in scaled_dataframes:
        input_layer = layers.Input(shape=(input_dim,))
        encoder = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoder = layers.Dense(input_dim, activation='linear')(encoder)

        autoencoder = tf.keras.Model(input_layer, decoder)

        # Compile and train the autoencoder
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(scaled_df, scaled_df, epochs=100, batch_size=32)

        # Add trained autoencoder to the list
        autoencoders.append(autoencoder)


    anomaly_threshold = []  # Adjust this threshold as needed
    threshold=[]

    anomalies = []

    # Iterate over each scaled dataframe and its corresponding autoencoder
    for scaled_df, autoencoder in zip(scaled_dataframes, autoencoders):
        # Reconstruct the data using the autoencoder
        reconstructed_data = autoencoder.predict(scaled_df)

        # Calculate the reconstruction error
        reconstruction_error = np.mean(np.square(scaled_df - reconstructed_data), axis=1)

        # Calculate the threshold dynamically
        threshold = np.percentile(reconstruction_error, 95)
        print(threshold)

        # Identify anomalies based on the threshold
        view_anomalies = scaled_df[reconstruction_error > threshold]

        # Add the anomalies to the list
        anomalies.append(view_anomalies)

    # List to store anomaly flags
    anomaly_flags = []

    # Iterate over each view and its corresponding autoencoder and scaled dataframe
    for autoencoder, scaled_df in zip(autoencoders, scaled_dataframes):
        # Get the last entry of the scaled dataframe
        last_entry = scaled_df[0]
        print (last_entry)

        # Reshape the last entry to match the autoencoder input shape
        last_entry = last_entry.reshape(1, -1)

        # Reconstruct the data using the autoencoder
        reconstructed_data = autoencoder.predict(last_entry)

        # Calculate the reconstruction error
        reconstruction_error = np.mean(np.square(last_entry - reconstructed_data))

        # Check if the reconstruction error exceeds the threshold for anomaly detection
        #anomaly_threshold = 0.01  # Adjust the threshold as needed
        is_anomaly = int(reconstruction_error > threshold)

        # Store the anomaly flag
        anomaly_flags.append(is_anomaly)

    # Establish connection to Snowflake
    conn = sf.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    warehouse=warehouse,
    database=snowflake_database,
    schema=snowflake_schema
    )

    from datetime import date
    current_date = date.today().strftime("%Y-%m-%d")


    # Iterate over the anomaly flags and insert them into the Snowflake table
    for index, flag in enumerate(anomaly_flags):
        meter_id = views[index]
        anomaly_flag_value = flag

        # Construct the SQL query
        query = f"INSERT INTO anomalies (meterId, anomaly_flag_value, date) VALUES ('{meter_id}', {anomaly_flag_value}, '{current_date}')"

        # Execute the SQL query
        cursor.execute(query)

        # Commit the changes
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()
        
    return 'Training completed successfully'


if __name__ == '__main__':
    app.run()
