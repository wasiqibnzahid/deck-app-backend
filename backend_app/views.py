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
    start = request.GET.get('start', "2020-01-01")
    end = request.GET.get('end', '2030-01-01')

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
#           FROM `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
#          LIMIT 50;
#     '''

    query = f"""
    SELECT
  IId as IId,
  SLat as SLat,
  SLong as SLong,
  ANY_VALUE(Title) as Title,
  ANY_VALUE(Description) as Description,
  ANY_VALUE(Image) as Image,
   AVG(weekly_sales) AS avg_weekly_sales,

  ST_Distance(ANY_VALUE(location),
ST_GEOGPOINT({longitude}, {latitude})
  ) AS distance
FROM
`geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
where date between DATE('{start}') AND DATE('{end}')
group by SLat, SLong, IId
ORDER BY
  distance
LIMIT
  50;
"""
    all_records = f'''
    select
  IId  as IId ,
  SLat  as SLat ,
  SLong  as SLong ,
  ANY_VALUE(Title)  as Title ,
  ANY_VALUE(Description)  as Description ,
  ANY_VALUE(Image)  as Image ,
  AVG(weekly_sales) AS avg_weekly_sales
  from `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
where date between DATE('{start}') AND DATE('{end}')

    group by SLat, SLong, IId
    '''

    query_job = client.query(query)
    all_items = client.query(all_records).result()
    results = query_job.result()

    results = [dict(row.items()) for row in results]
    all_items = [dict(row.items()) for row in all_items]
    # all_items = []
    return JsonResponse({"closest": results, "all": all_items}, safe=False)


def get_text_embedding(text: str) -> np.ndarray:
    tokens = openclip_tokenizer([text])
    with torch.no_grad():
        embedding = openclip_model.encode_text(tokens)
    return embedding.cpu().numpy().flatten()


@require_GET
def search_description(request):
    search_text = request.GET.get('search_text')
    longitude = request.GET.get('longitude')
    latitude = request.GET.get('latitude')
    mode = request.GET.get("mode")  # "bigquery" or "model" or "id"
    start = request.GET.get("start")
    end = request.GET.get("end")

    if not search_text:
        return JsonResponse({'error': 'Search text parameter is required.'}, status=400)

    # Initialize BigQuery client

    client = bigquery.Client()
    count = 0

    if mode == "bigquery":
        # Case-insensitive text search in BigQuery
        print("HEERE RNONO NOW")
        query = f'''
            SELECT  ANY_VALUE(HId) as HId ,
                    IId as IId ,
                    ANY_VALUE(HPId) as HPId ,
                    ANY_VALUE(C0) as C0 ,
                    ANY_VALUE(C1) as C1 ,
                    ANY_VALUE(C2) as C2 ,
                    SLat as SLat ,
                    SLong as SLong ,
                    ANY_VALUE(SPId) as SPId ,
                    ANY_VALUE(D2Sm) as D2Sm ,
                    ANY_VALUE(HIdDensity) as HIdDensity ,
                    ANY_VALUE(Title) as Title ,
                    ANY_VALUE(Description) as Description ,
                    ANY_VALUE(Image) as Image ,
                    ANY_VALUE(weeknumber) as weeknumber ,
                    ANY_VALUE(weekly_sales) as weekly_sales ,
               ST_Distance(
     ANY_VALUE(location),
     ST_GEOGPOINT({longitude}, {latitude})
   ) AS distance
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
            WHERE date between DATE('{start}') AND DATE('{end}') and LOWER(Description) LIKE LOWER('%{search_text}%')
            group by SLat, SLong, IId
            order by distance
            LIMIT 50
        '''
        query_job = client.query(query)
        result = query_job.result()

        # Convert results to JSON format
        result = [dict(row.items()) for row in result]

        return JsonResponse(result, safe=False)

    elif (mode == 'model'):
        # Generate embedding for the search text
        search_text_embedding = get_text_embedding(search_text)

        # Define query to fetch embeddings from BigQuery
        query = f'''
            SELECT IId, ANY_VALUE(image_embedding) as image_embedding
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
            where date between DATE('{start}') AND DATE('{end}')

            group by IId, SLat, SLong
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
            SELECT
                    IId,
                    SLat,
                    SLong,
                    ANY_VALUE(Title),
                    ANY_VALUE(Description),
                    ANY_VALUE(Image)
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`
            where date between DATE('{start}') AND DATE('{end}')
            and IId IN ({ids_placeholder})
            group by IId, SLat, SLong

        '''

        full_query_job = client.query(full_query)
        result = full_query_job.result()

        # Convert results to JSON format
        result = [dict(row.items()) for row in result]

        return JsonResponse(result, safe=False)
    else:
        query = f'''
            SELECT  ANY_VALUE(HId) as HId,
                    IId as IId,
                    ANY_VALUE(C0) as C0,
                    ANY_VALUE(C1) as C1,
                    ANY_VALUE(C2) as C2,
                    SLat as SLat,
                    SLong as SLong,
                    ANY_VALUE(SPId) as SPId,
                    ANY_VALUE(D2Sm) as D2Sm,
                    ANY_VALUE(HIdDensity) as HIdDensity,
                    ANY_VALUE(Title) as Title,
                    ANY_VALUE(Description) as Description,
                    ANY_VALUE(Image) as Image,
                    ANY_VALUE(weeknumber) as weeknumber,
                    ANY_VALUE(weekly_sales) as weekly_sales,
                    ARRAY_AGG(STRUCT(date, forecast, weekly_sales) ORDER BY date) AS forecast_records,
               ST_Distance(
     ANY_VALUE(location),
     ST_GEOGPOINT({longitude}, {latitude})
   ) AS distance
            FROM `geosearch-1511586674493.geoAppDB1.geospatialSales_new_final`

            WHERE IId = {search_text}
            and date between DATE('{start}') AND DATE('{end}')

            group by SLat, SLong, IId
            order by distance
            LIMIT 50
        '''
        query_job = client.query(query)
        result = query_job.result()

        # Convert results to JSON format
        result = [dict(row.items()) for row in result]
        return JsonResponse(result, safe=False)
