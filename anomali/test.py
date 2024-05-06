import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import snowflake.connector as sf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define the S3 bucket and Snowflake parameters
snowflake_account = 'at67872.ap-south-1.aws'
snowflake_user = 'FYP4'
snowflake_password = '!Q2w3e4r5t6y'
warehouse='COMPUTE_WH'
snowflake_database = 'COAPDATA'
snowflake_schema = 'METER_SCHEMA'

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

view_dataframes = {}

# Iterate over each water meter view
for view in views:
    # Retrieve the data from the current water meter view
    query = f"SELECT Date, total FROM {snowflake_database}.{snowflake_schema}.{view} ORDER BY Date DESC"
    df = pd.read_sql(query, conn)
        
    # # Scale the data using standardization
    scaler = StandardScaler()
    # Convert the 'total' column to a numpy array
    data = df['TOTAL'].values 
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
    # # Store the dataframe in the dictionary
    view_dataframes[view] = df
    
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

autoencoder.fit(scaled_data, scaled_data,
                epochs=50,
                batch_size=32)


# Use the autoencoder model to make predictions
predictions = autoencoder.predict(scaled_data)

# Calculate the reconstruction error
mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)
anomalies = np.where(mse > threshold)[0]

