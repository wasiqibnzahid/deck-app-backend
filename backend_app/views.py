from google.cloud import bigquery
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
import torch

openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')


@require_GET
def get_closest_records(request):
    longitude = request.GET.get('longitude')
    latitude = request.GET.get('latitude')

    if not longitude or not latitude:
        return JsonResponse({'error': 'Longitude and Latitude parameters are required.'}, status=400)

    # Initialize BigQuery client
    client = bigquery.Client()

#     query = f'''
#         SELECT  HId,
#   IId,
#   HPId,
#   C0,
#   C1,
#   C2,
#   SLat,
#   SLong,
#   SPId,
#   D2Sm,
#   HIdDensity,
#   Title,
#   Description,
#   Image,
#   weeknumber,
#   weekly_sales,
#                ST_DISTANCE(ST_GeogPoint(SLong, SLat), ST_GeogPoint({longitude}, {latitude})) AS distance
#           FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
#          LIMIT 50;
#     '''

    query = f"""
    SELECT
  HId,
  IId,
  HPId,
  C0,
  C1,
  C2,
  SLat,
  SLong,
  SPId,
  D2Sm,
  HIdDensity,
  Title,
  Description,
  Image,
  weeknumber,
  weekly_sales,
  ST_Distance(
    ST_GEOGPOINT(SLong, SLat),
    ST_GEOGPOINT({longitude}, {latitude})
  ) AS distance
FROM
`geosearch-1511586674493.geoAppDB1.geospatialSales`
ORDER BY
  distance
LIMIT
  50;
"""
    all_records = f'''
    select   HId,
  IId,
  HPId,
  C0,
  C1,
  C2,
  SLat,
  SLong,
  SPId,
  D2Sm,
  HIdDensity,
  Title,
  Description,
  Image,
  weeknumber,
  weekly_sales from `geosearch-1511586674493.geoAppDB1.geospatialSales`
    '''

    query_job = client.query(query)
    all_items = client.query(all_records).result()
    results = query_job.result()

    results = [dict(row.items()) for row in results]
    all_items = [dict(row.items()) for row in all_items]
    return JsonResponse({"closest": results, "all": all_items}, safe=False)


def get_text_embedding(text: str) -> np.ndarray:
    tokens = openclip_tokenizer([text])
    with torch.no_grad():
        embedding = openclip_model.encode_text(tokens)
    return embedding.cpu().numpy().flatten()


@require_GET
def search_description(request):
    search_text = request.GET.get('search_text')
    mode = request.GET.get("mode")  # "bigquery" or "model"

    if not search_text:
        return JsonResponse({'error': 'Search text parameter is required.'}, status=400)

    # Initialize BigQuery client

    client = bigquery.Client()
    count = 0
    if mode == "bigquery":
        # Case-insensitive text search in BigQuery
        print("HEERE RNONO NOW")
        query = f'''
            SELECT  HId,
                    IId,
                    HPId,
                    C0,
                    C1,
                    C2,
                    SLat,
                    SLong,
                    SPId,
                    D2Sm,
                    HIdDensity,
                    Title,
                    Description,
                    Image,
                    weeknumber,
                    weekly_sales
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
            WHERE LOWER(Description) LIKE LOWER('%{search_text}%')
            LIMIT 50
        '''
        query_job = client.query(query)
        result = query_job.result()

        # Convert results to JSON format
        result = [dict(row.items()) for row in result]

        return JsonResponse(result, safe=False)

    else:
        # Generate embedding for the search text
        search_text_embedding = get_text_embedding(search_text)

        # Define query to fetch embeddings from BigQuery
        query = '''
            SELECT IId, image_embedding
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
        '''
        query_job = client.query(query)
        rows = query_job.result()

        similarities = []
        for row in rows:
            record_id = row['IId']
            stored_image_embedding = np.array(row['image_embedding'])
            if stored_image_embedding.size == 0 or search_text_embedding.size == 0:
                continue  # Skip this record if embeddings are empty

            # Compute cosine similarity for image embeddings
            image_similarity = cosine_similarity(
                [search_text_embedding], [stored_image_embedding])[0][0]

            similarities.append((record_id, image_similarity))
            count += 1
            print(f"Done for {count}")

        # Sort by similarity and get top K records
        top_records = sorted(
            similarities, key=lambda x: x[1], reverse=True)[:50]

        # Fetch the full records based on top K results
        ids = [record[0] for record in top_records]
        # Use map to convert integers to strings without quotes
        ids_placeholder = ', '.join(map(str, ids))

        full_query = f'''
            SELECT  HId,
                    IId,
                    HPId,
                    C0,
                    C1,
                    C2,
                    SLat,
                    SLong,
                    SPId,
                    D2Sm,
                    HIdDensity,
                    Title,
                    Description,
                    Image,
                    weeknumber,
                    weekly_sales
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
            WHERE IId IN ({ids_placeholder})
        '''

        full_query_job = client.query(full_query)
        result = full_query_job.result()

        # Convert results to JSON format
        result = [dict(row.items()) for row in result]

        return JsonResponse(result, safe=False)

# import torch
# from torchvision.transforms import Resize, Compose, Normalize, ToTensor
# import clip
# import io
# from PIL import Image
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from django.http import JsonResponse

# def search_description(request):
#     search_text = request.GET.get('search_text')

#     if not search_text:
#         return JsonResponse({'error': 'Search text parameter is required.'}, status=400)

#     # Initialize BigQuery client
#     client = bigquery.Client()

#     # Generate embedding for the search text
#     search_text_embedding = get_text_embedding(search_text)

#     # Define query to fetch embeddings from BigQuery
#     query = '''
#         SELECT IId, text_embedding, image_embedding
#         FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
#     '''
#     query_job = client.query(query)
#     rows = query_job.result()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)

#     def preprocess_image(image_bytes):
#         image = Image.open(io.BytesIO(image_bytes))
#         image_input = preprocess(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             image_features = model.encode_image(image_input)
#         return image_features

#     def calculate_image_similarity(search_text_embedding, image_embedding):
#         image_features = torch.tensor(image_embedding, device=device).unsqueeze(0)
#         text_features = torch.tensor(search_text_embedding, device=device).unsqueeze(0)
#         similarity = image_features @ text_features.T
#         return similarity.item()

#     similarities = []
#     for row in rows:
#         record_id = row['IId']
#         stored_text_embedding = np.array(row['text_embedding'])
#         stored_image_embedding = np.array(row['image_embedding'])
#         if stored_text_embedding.size == 0 or search_text_embedding.size == 0:
#             continue

#         text_similarity = cosine_similarity(
#             [search_text_embedding], [stored_text_embedding])[0][0]

#         image_bytes = row['image']  # Assuming image is stored as bytes
#         image_features = preprocess_image(image_bytes)
#         image_similarity = calculate_image_similarity(search_text_embedding, image_features)

#         combined_similarity = 0.5 * text_similarity + 0.5 * image_similarity
#         similarities.append((record_id, combined_similarity))

#     top_records = sorted(
#         similarities, key=lambda x: x[1], reverse=True)[:50]

#     ids = [record[0] for record in top_records]
#     ids_placeholder = ', '.join(map(str, ids))

#     full_query = f'''
#         SELECT  HId,
#                 IId,
#                 HPId,
#                 C0,
#                 C1,
#                 C2,
#                 SLat,
#                 SLong,
#                 SPId,
#                 D2Sm,
#                 HIdDensity,
#                 Title,
#                 Description,
#                 Image,
#                 weeknumber,
#                 weekly_sales
#         FROM `geosearch-1511586674493.geoAppDB1.geospatialSales`
#         WHERE IId IN ({ids_placeholder})
#     '''

#     full_query_job = client.query(full_query)
#     result = full_query_job.result()

#     result = [dict(row.items()) for row in result]

#     return JsonResponse(result, safe=False)


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
