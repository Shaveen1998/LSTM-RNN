import snowflake.connector as sf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Define the S3 bucket and Snowflake parameters
s3_bucket = 'fypmlbucket'
snowflake_account = 'ND64012.ap-south-1.aws'
snowflake_user = 'FYP'
snowflake_password = '$Tarwars123'
snowflake_database = 'COAPDATA'
snowflake_schema = 'PUBLIC'
snowflake_table = 'mlview'

#   account: 'BF48652.ap-south-1.aws',
#   username: 'FYP',
#   password: '!Q2w3e4r5t',
#   database: 'COAPDATA',
#   schema: 'PUBLIC',
#   warehouse: 'COMPUTE_WH',



# Define the model parameters
n_input = 10
n_features = 1
model_file_name = 'model.h5'
# load the initial LSTM model
model = load_model(model_file_name)

# Define the main Lambda function handler



def train():    
    # Connect to Snowflake and retrieve the latest data
    conn = sf.connect(user=snowflake_user, password=snowflake_password, account=snowflake_account)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {snowflake_database}.{snowflake_schema}.{snowflake_table} ORDER BY Date DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(data, columns=['Date', 'consumption'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Scale the data using the same scaler used for training
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    # Update the model with the new data
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
    
    # take the last 10 and reshape
    last_scaled_batch = scaled_data[-10:]
    last_scaled_batch = last_scaled_batch.reshape((1, n_input, n_features))
    
    
    model.fit(generator, epochs=1, verbose=0)
    
    prediction = model.predict(last_scaled_batch)
    true_prediction = scaler.inverse_transform(prediction)[0][0]
    
    snowflake_conn = sf.connect(
        user='FYP',
        password='$Tarwars123',
        account='ND64012.ap-south-1.aws',
        warehouse='COMPUTE_WH',
        database='COAPDATA',
        schema='PUBLIC'
    )

    
    
    
    cursor = snowflake_conn.cursor()
    cursor.execute(f"INSERT INTO predict_table (date, prediction) VALUES (DATEADD(DAY, 1, (SELECT MAX(date) FROM mlview)), {true_prediction})")
    cursor.close()
    snowflake_conn.close()
    
    return 'Training completed successfully'





