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

# Define the model parameters
n_input = 10
n_features = 1
model_file_name = 'model.h5'

# Define the main Lambda function handler
def lambda_handler(event, context):
    
    # Load the saved model from S3
    s3 = boto3.client('s3')
    s3_object = s3.get_object(Bucket=s3_bucket, Key=model_file_name)
    model_content = s3_object['Body'].read()
    model = load_model(model_content)

    # Connect to Snowflake and retrieve the latest data
    conn = sf.connect(user=snowflake_user, password=snowflake_password, account=snowflake_account)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {snowflake_database}.{snowflake_schema}.{snowflake_table} ORDER BY Date DESC LIMIT {n_input}")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(data, columns=['Date', 'Production'])
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

    # Save the updated model back to S3
    s3.put_object(Bucket=s3_bucket, Key=model_file_name, Body=model.to_bytes())
    
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
