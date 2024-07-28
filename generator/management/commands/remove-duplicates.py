import pandas as pd
from django.core.management.base import BaseCommand, CommandError

# Load the data from the file


def removeDuplicates():
    file_path = '/home/wasiq/out.csv'  # Update this with the correct file path
    df = pd.read_csv(file_path)

    # Remove duplicates based on 'SLat' and 'SLong'
    df_no_duplicates = df.drop_duplicates(subset=['SLat', 'SLong'])

    # Save the cleaned data to a new file
    # Update this with the desired output path
    output_path = '/home/wasiq/out2.csv'
    df_no_duplicates.to_csv(output_path, index=False)

    print(f"Removed duplicates and saved cleaned data to {output_path}")


class Command(BaseCommand):
    def handle(self, *args, **options):
        removeDuplicates()
    