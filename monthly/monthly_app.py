import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras.preprocessing.sequence import TimeseriesGenerator
from flask import Flask
import snowflake.connector as sf

#Define the S3 bucket and Snowflake parameters
snowflake_account = 'at67872.ap-south-1.aws'
snowflake_user = 'FYP4'
snowflake_password = '!Q2w3e4r5t6y'
snowflake_database = 'COAPDATA'
snowflake_schema = 'PUBLIC'
snowflake_warehouse = 'COMPUTE_WH'
snowflake_table = 'monthly_consumption'
app = Flask(__name__)

# Define the model parameters
n_input = 12
n_features = 1
model_file_name = 'monthly_model.h5'
#load the initial LSTM model
model = load_model(model_file_name)

# Define the main Lambda function handler


@app.route('/train_monthly', methods=['POST'])
def train():    
    #Connect to Snowflake and retrieve the latest data
    conn = sf.connect(user=snowflake_user, password=snowflake_password, account=snowflake_account)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {snowflake_database}.{snowflake_schema}.{snowflake_table} ORDER BY date DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(data, columns=['Month', 'consumption'])
    df.set_index('Month', inplace=True)
    df.sort_index(inplace=True)
    
    # Scale the data using the same scaler used for training
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    # Update the model with the new data
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
    
    # take the last 10 and reshape
    last_scaled_batch = scaled_data[-12:]
    last_scaled_batch = last_scaled_batch.reshape((1, n_input, n_features))
    
    
    model.fit(generator, epochs=1, verbose=0)
    model.save('monthly_model.h5')
    
    prediction = model.predict(last_scaled_batch)
    true_prediction = scaler.inverse_transform(prediction)[0][0]
    
    snowflake_conn = sf.connect(
            user='FYP4',
            password='!Q2w3e4r5t6y',
            account='at67872.ap-south-1.aws',
            warehouse='COMPUTE_WH',
            database='COAPDATA',
            schema='PUBLIC'
    )


    cursor = snowflake_conn.cursor()
    cursor.execute(f"INSERT INTO monthly_predict_table (date, prediction) VALUES (DATEADD(MONTH, 1, (SELECT MAX(date) FROM monthly_consumption)), {true_prediction})")
    cursor.close()
    snowflake_conn.close()
    
    return 'Training completed successfully'


if __name__ == '__main__':
    app.run()














