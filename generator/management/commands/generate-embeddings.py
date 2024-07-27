import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from google.cloud import bigquery
from transformers import BertTokenizer, TFBertModel
import aiohttp
import asyncio
import concurrent.futures

# Initialize BigQuery client
client = bigquery.Client()

# Load pre-trained model for image embeddings
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load pre-trained model and tokenizer for text embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_image_embedding(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array)
    return embedding.flatten().tolist()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
    outputs = text_model(inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding.tolist()

async def fetch_and_update_record(semaphore, record):
    async with semaphore:
        image_url = record['Image']
        description = record['Description']
        
        loop = asyncio.get_event_loop()
        print(f"generating embedding for image {image_url}")
        image_embedding = await loop.run_in_executor(None, get_image_embedding, image_url)
        print(f"generating embedding for text {description}")
        text_embedding = await loop.run_in_executor(None, get_text_embedding, description)
        
        image_embedding_str = ', '.join(map(str, image_embedding))
        text_embedding_str = ', '.join(map(str, text_embedding))
        
        # Update BigQuery table
        print(f"updating query for image {image_url}")
        update_query = f'''
            UPDATE `geoapps-429420.New2Geo.geoserver`
            SET image_embedding = [{image_embedding_str}],
                text_embedding = [{text_embedding_str}]
            WHERE SLat = {record['SLat']} AND SLong = {record['SLong']};
        '''
        client.query(update_query)
        print("done for image {image_url}")

async def fetch_and_update_records():
    query = '''
        SELECT *
        FROM `geoapps-429420.New2Geo.geoserver`
        where Description like '% Tall coat by New LookThat new-coat feeling Notch lapelsPress-stud fastening Side pockets Regular fit Product Code: 113448586 }; { Brand :  Since setting up shop i%'
        ;
    '''
    query_job = client.query(query)
    rows = query_job.result()
    
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks
    tasks = []
    for row in rows:
        record = dict(row)
        task = fetch_and_update_record(semaphore, record)
        tasks.append(task)
    
    await asyncio.gather(*tasks)

# Run the function
asyncio.run(fetch_and_update_records())
