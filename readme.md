# Run Project
Install packages using
```
pip install -r requirements.txt
python manage.py runserver
```

# Set key path
You can set key path in settings.py
Change this variable
```
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/wasiq/personal/back/key.json'
```
To the actual path of the key


# This runs on default on PORT 8000

The views.py has a get function, you can add new models there
There is a file in commands folder to generate embeddings for images, you should use it to pre-create embeddings
instead of generating them in runtime, you can modify it to use any other model instead of OpenClip
