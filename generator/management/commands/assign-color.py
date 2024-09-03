import os
from django.core.management.base import BaseCommand
from google.cloud import bigquery

class Command(BaseCommand):
    help = 'Update the sales colors based on weekly sales data.'

    def handle(self, *args, **kwargs):
        # Initialize BigQuery client
        client = bigquery.Client()

        # Replace with your actual BigQuery table reference
        table_id = "geosearch-1511586674493.geoAppDB1.geospatialSales_new_final"

        # Step 1: Get the maximum weekly sales value
        max_sales_query = f"""
        SELECT MAX(weekly_sales) AS max_sales
        FROM `{table_id}`
        """
        query_job = client.query(max_sales_query)
        results = query_job.result()

        max_sales = None
        for row in results:
            max_sales = row.max_sales
        print(f"MAX SALES GOTTEN {max_sales}")

        if max_sales is None:
            self.stdout.write(self.style.ERROR('No sales data found.'))
            return

        # Step 2: Calculate sales ranges
        high_threshold = max_sales * 0.7
        medium_threshold = max_sales * 0.3

        # Step 3: Update colors based on sales ranges
        update_color_query = f"""
        UPDATE `{table_id}`
        SET color = CASE
            WHEN weekly_sales >= {high_threshold} THEN [0, 255, 0]  -- Green for high
            WHEN weekly_sales >= {medium_threshold} THEN [255, 255, 0]  -- Yellow for medium
            ELSE [255, 0, 0]  -- Red for low
        END
        WHERE TRUE
        """
        update_job = client.query(update_color_query)
        update_job.result()  # Wait for the job to complete

        self.stdout.write(self.style.SUCCESS('Sales colors updated successfully.'))
