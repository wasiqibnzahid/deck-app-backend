from google.cloud import bigquery
from django.http import JsonResponse
from django.views.decorators.http import require_GET


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

    if not search_text:
        return JsonResponse({'error': 'Search text parameter is required.'}, status=400)

    # Initialize BigQuery client
    client = bigquery.Client()

    query = f'''
           SELECT *
             FROM `geoapps-429420.New2Geo.geoserver`
            WHERE LOWER(Description) LIKE LOWER('%{search_text}%')
               OR LOWER(Title) LIKE LOWER('%{search_text}%')
            LIMIT 50;
       '''

    query_job = client.query(query)
    results = query_job.result()

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
