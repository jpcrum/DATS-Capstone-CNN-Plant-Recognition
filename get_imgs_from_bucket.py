bucket_name = 'dats-capstone-bucket'
prefix = 'Black-grass/'
dl_dir = 'C:/Users/sjcrum/Documents/GitHub/Plant-Image-Recognition/DATS-Capstone-CNN-Plant-Recognition/Documentation/'

storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name=bucket_name)
blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
for blob in blobs:
    filename = blob.name.replace('/', '_')
    blob.download_to_filename(dl_dir + filename)  # Download
