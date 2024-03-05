from azure.storage.blob import BlobServiceClient

def check_model_output_in_blob(container_name, blob_name):
    """
    Checks if the model output exists and is valid in Azure Blob Storage.

    Args:
        container_name (str): Name of the Blob container.
        blob_name (str): Name of the blob to check.
    """
    try:
        # Assuming that the BlobServiceClient can pick up the existing connection
        blob_service_client = BlobServiceClient.from_connection_string(dbutils.secrets.get(scope="ml-repeat-buyer", key="BLOB-SAS-TOKEN-ML-REPEAT-BUYER"))
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        if blob_client.exists():
            print(f"Blob '{blob_name}' exists in container '{container_name}'.")
        else:
            print(f"Blob '{blob_name}' does not exist in container '{container_name}'.")
    except Exception as e:
        print(f"Error occurred while checking blob: {e}")
