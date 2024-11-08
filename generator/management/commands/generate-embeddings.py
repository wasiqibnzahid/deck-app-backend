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
import asyncio
from django.core.management.base import BaseCommand
import torch
import open_clip
import json
import os

# Initialize BigQuery client
client = bigquery.Client()
processed_count = 0
lock = asyncio.Lock()

# Load pre-trained model for image embeddings
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load pre-trained model and tokenizer for text embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = TFBertModel.from_pretrained('bert-base-uncased')

openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')


def get_image_embedding(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        # Preprocess the image and add batch dimension
        img = openclip_preprocess(img).unsqueeze(0)

        with torch.no_grad():
            embedding = openclip_model.encode_image(img)

        return embedding.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', max_length=512,
                       truncation=True, padding='max_length')
    outputs = text_model(inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding.tolist()


async def fetch_and_update_record(semaphore, record_batch):
    async with semaphore:
        loop = asyncio.get_event_loop()

        embeddings = []
        print("STARTING A BATCH OF LENGTH ", len(record_batch))
        for record in record_batch:
            image_url = record['Image']
            description = record['Description']

            image_embedding = await loop.run_in_executor(None, get_image_embedding, image_url)
            text_embedding = await loop.run_in_executor(None, get_text_embedding, description)

            embeddings.append((record, image_embedding, text_embedding))

            async with lock:
                global processed_count
                processed_count += 1
                print(f'Processed {processed_count} records')

        print("Generating update query")
        update_query = f"UPDATE `{os.getenv('BIGQUERY_TABLE_NAME')}` SET image_embedding = CASE "
        for record, image_embedding, _ in embeddings:
            image_embedding_str = ', '.join(map(str, image_embedding))
            update_query += f"WHEN SLat = {record['SLat']} AND SLong = {
                record['SLong']} THEN [{image_embedding_str}] "

        update_query += "END, text_embeddin}g = CASE "
        for record, _, text_embedding in embeddings:
            text_embedding_str = ', '.join(map(str, text_embedding))
            update_query += f"WHEN SLat = {record['SLat']} AND SLong = {
                record['SLong']} THEN [{text_embedding_str}] "

        # Adding WHERE clause to ensure valid update
        update_query += "END WHERE SLat IN ("
        update_query += ', '.join([str(record['SLat'])
                                  for record, _, _ in embeddings])
        update_query += ") AND SLong IN ("
        update_query += ', '.join([str(record['SLong'])
                                  for record, _, _ in embeddings])
        update_query += ");"

        print(f"The query is {update_query}")
        client.query(update_query)
        print("DONE A BATCH")


async def fetch_and_update_records():
    query = f'''
        SELECT *
        FROM `{os.getenv('BIGQUERY_TABLE_NAME')}`;
    '''

    query_job = client.query(query)
    rows = query_job.result()

    semaphore = asyncio.Semaphore(20)  # Limit to 10 concurrent tasks
    tasks = []
    record_batch = []
    batch_size = 5

    for row in rows:
        print(f"URL IS {row["Image"]}")
        record_batch.append(dict(row))
        if len(record_batch) == batch_size:
            task = fetch_and_update_record(semaphore, record_batch)
            tasks.append(task)
            record_batch = []

    if record_batch:
        task = fetch_and_update_record(semaphore, record_batch)
        tasks.append(task)

    await asyncio.gather(*tasks)


class Command(BaseCommand):
    def handle(self, *args, **options):
        asyncio.run(fetch_and_update_records())
        print(f'Total records processed: {processed_count}')
