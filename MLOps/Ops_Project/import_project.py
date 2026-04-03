from google.cloud import storage


def list_code_files(bucket_name, prefix=None):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # list_blobs is a generator; it doesn't load everything at once
    blobs = bucket.list_blobs(prefix=prefix)
    print(blobs)
    extensions = ('.py')
    
    for blob in blobs:
        if blob.name.endswith(extensions):
            print(f"Found: {blob.name}")
            

list_code_files("codebases-03-26")