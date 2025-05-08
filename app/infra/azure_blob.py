import os
from azure.storage.blob import BlobServiceClient, ContentSettings

connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

def upload_blob(container_name, blob_name, data, content_type):
    container_client = blob_service_client.get_container_client(container_name)
    # Ensure the container exists
    try:
        container_client.create_container()
    except Exception:
        pass  # Container likely already exists
    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

def download_blob(container_name, blob_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob()
    return stream.readall() 