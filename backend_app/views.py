from google.cloud import bigquery
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = TFBertModel.from_pretrained('bert-base-uncased')


def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors='tf',
                            max_length=512, truncation=True, padding='max_length')
    outputs = text_model(inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding


@require_GET
def get_closest_records(request):
    longitude = request.GET.get('longitude')
    latitude = request.GET.get('latitude')

    if not longitude or not latitude:
        return JsonResponse({'error': 'Longitude and Latitude parameters are required.'}, status=400)

    # Initialize BigQuery client
    client = bigquery.Client()

    query = f'''
        SELECT *,
               ST_DISTANCE(ST_GeogPoint(SLong, SLat), ST_GeogPoint({longitude}, {latitude})) AS distance
          FROM `geoapps-429420.New2Geo.geoserver`
         LIMIT 50;
    '''

    query_job = client.query(query)
    results = query_job.result()

    # Convert results to JSON format
    results = [dict(row.items()) for row in results]

    return JsonResponse(results, safe=False)


@require_GET
def search_description(request):
    search_text = request.GET.get('search_text')

    # if not search_text:
    #     return JsonResponse({'error': 'Search text parameter is required.'}, status=400)

    # # Initialize BigQuery client
    client = bigquery.Client()

    # query = f'''
    #        SELECT *
    #          FROM `geoapps-429420.New2Geo.geoserver`
    #         WHERE LOWER(Description) LIKE LOWER('%{search_text}%')
    #            OR LOWER(Title) LIKE LOWER('%{search_text}%')
    #         LIMIT 50;
    #    '''

    # query_job = client.query(query)
    # results = query_job.result()
    # Generate embedding for the search text
    search_text_embedding = get_text_embedding(search_text)

    # Define query to fetch embeddings from BigQuery
    query = '''
        SELECT id, text_embedding, image_embedding
        FROM `your_project.your_dataset.your_table`
    '''
    query_job = client.query(query)
    rows = query_job.result()

    similarities = []
    for row in rows:
        record_id = row['id']
        stored_text_embedding = np.array(row['text_embedding'])
        stored_image_embedding = np.array(row['image_embedding'])

        # Compute cosine similarity for text embeddings
        text_similarity = cosine_similarity(
            [search_text_embedding], [stored_text_embedding])[0][0]

        # Optionally, compute similarity for image embeddings if needed
        # For example, if you want to use a constant weight for image similarity
        # search_image_embedding = get_image_embedding(search_image_url)
        # image_similarity = cosine_similarity([search_image_embedding], [stored_image_embedding])[0][0]

        # Combine similarities if using image embeddings
        # combined_similarity = (text_similarity + image_similarity) / 2
        # Or just use text similarity if focusing on text
        combined_similarity = text_similarity

        similarities.append((record_id, combined_similarity))

    # Sort by similarity and get top K records
    top_records = sorted(
        similarities, key=lambda x: x[1], reverse=True)[:50]

    # Fetch the full records based on top K results
    ids = [record[0] for record in top_records]
    ids_placeholder = ', '.join([f"'{id}'" for id in ids])
    full_query = f'''
        SELECT *
        FROM `your_project.your_dataset.your_table`
        WHERE id IN ({ids_placeholder})
    '''
    full_query_job = client.query(full_query)
    result = full_query_job.result()

    return result

    # Convert results to JSON format
    results = [dict(row.items()) for row in results]

    return JsonResponse(results, safe=False)


@require_GET
def get_points_within_range(request):
    min_longitude = request.GET.get('min_longitude')
    max_longitude = request.GET.get('max_longitude')
    min_latitude = request.GET.get('min_latitude')
    max_latitude = request.GET.get('max_latitude')

    if not min_longitude or not max_longitude or not min_latitude or not max_latitude:
        return JsonResponse({'error': 'min_longitude, max_longitude, min_latitude, and max_latitude parameters are required.'}, status=400)

    query = '''
        SELECT *
          FROM your_table_name
         WHERE SLong BETWEEN %s AND %s
           AND SLat BETWEEN %s AND %s
           LIMIT 50;
    '''

    with connection.cursor() as cursor:
        cursor.execute(
            query, [min_longitude, max_longitude, min_latitude, max_latitude])
        columns = [col[0] for col in cursor.description]
        results = [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

    return JsonResponse(results, safe=False)
