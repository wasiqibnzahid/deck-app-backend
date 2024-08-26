import asyncio
import random
from datetime import datetime, timedelta
from google.cloud import bigquery
from django.core.management.base import BaseCommand

# Initialize BigQuery client
client = bigquery.Client()

# Generate a random date within a range


def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

# Generate random data for the specified fields


def generate_random_data(record):
    record_copy = record.copy()

    # Randomize specific fields
    record_copy['forecast'] = random.randint(0, 100)
    record_copy['weekly_sales'] = round(random.uniform(100, 1000), 2)
    record_copy['date'] = random_date(
        datetime.now() - timedelta(days=365), datetime.now()).strftime('%Y-%m-%d')

    # Optional: Randomize other fields if desired
    # For demonstration, we'll set default values or randomize some fields
    record_copy['SLat'] = random.uniform(-90.0, 90.0)  # Random latitude
    record_copy['SLong'] = random.uniform(-180.0, 180.0)  # Random longitude
    record_copy['text_embedding'] = [random.uniform(
        0, 1) for _ in range(10)]  # Example for repeated field
    record_copy['image_embedding'] = [random.uniform(
        0, 1) for _ in range(10)]  # Example for repeated field

    return record_copy


async def randomize_data(batch_size=5):
    query = '''
        SELECT *
        FROM `geosearch-1511586674493.geoAppDB1.geospatialSales` LIMIT 10;
    '''

    query_job = client.query(query)
    rows = query_job.result()

    all_rows = []

    for row in rows:
        # Convert row to dict
        original_record = dict(row)

        # Randomize the original record
        randomized_original = generate_random_data(original_record)
        all_rows.append(randomized_original)

        # Create two duplicates with random data
        for _ in range(2):
            new_row = generate_random_data(original_record)
            all_rows.append(new_row)

    # Upload data in batches using SQL INSERT queries
    table_id = "geosearch-1511586674493.geoAppDB1.geospatialSales"
    total_records = len(all_rows)
    for start in range(0, total_records, batch_size):
        end = min(start + batch_size, total_records)
        batch_records = all_rows[start:end]

        # Create an INSERT SQL query
        insert_query = f"""
        INSERT INTO `{table_id}` (HId, IId, HPId, C0, C1, C2, SLat, SLong, SPId, D2Sm, HIdDensity, Title, Description, Image, weeknumber, weekly_sales, text_embedding, image_embedding, date, forecast)
        VALUES
        """
        values = []
        for record in batch_records:
            text_embedding = ','.join(
                map(str, record.get('text_embedding', [])))
            image_embedding = ','.join(
                map(str, record.get('image_embedding', [])))
            values.append(f"({record.get('HId', 'NULL')}, {record.get('IId', 'NULL')}, '{record.get('HPId', '')}', {record.get('C0', 'NULL')}, {record.get('C1', 'NULL')}, {record.get('C2', 'NULL')}, {record.get('SLat', 'NULL')}, {record.get('SLong', 'NULL')}, '{record.get('SPId', '')}', {record.get('D2Sm', 'NULL')}, {
                          record.get('HIdDensity', 'NULL')}, '{record.get('Title', '')}', '{record.get('Description', '')}', '{record.get('Image', '')}', {record.get('weeknumber', 'NULL')}, {record.get('weekly_sales', 'NULL')}, ARRAY[{text_embedding}], ARRAY[{image_embedding}], '{record.get('date', '')}', {record.get('forecast', 'NULL')})")
        insert_query += ', '.join(values)

        # Execute the INSERT query
        query_job = client.query(insert_query)
        query_job.result()  # Wait for the job to complete

        print(f"Inserted records from {start} to {
              end} ({end - start} records)")

    print(f"Total records inserted: {total_records}")


class Command(BaseCommand):
    def handle(self, *args, **options):
        asyncio.run(randomize_data())
        print('Data fetching, randomization, and duplication completed.')
