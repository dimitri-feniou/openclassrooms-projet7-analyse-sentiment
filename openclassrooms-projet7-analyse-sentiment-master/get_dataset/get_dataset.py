# Script for download and get dataset CSV
import os 
import requests
import zipfile

# URL file to download
url = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip"
# Folder to stock files
dataset_folder = "dataset"
zip_file_path = "sentiment140.zip"
csv_file_name = "training.1600000.processed.noemoticon.csv"

# Create folder 'dataset' if don't exist
os.makedirs(dataset_folder, exist_ok=True)

# Verify if CSV file exist
csv_path = os.path.join(dataset_folder, csv_file_name)
if os.path.exists(csv_path):
    print(f"Le fichier CSV existe déjà : {csv_path}")
else:
    # Download zip file if don't exist
    if not os.path.exists(zip_file_path):
        print("Téléchargement du fichier ZIP...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Fichier ZIP téléchargé : {zip_file_path}")
        else:
            print("Erreur lors du téléchargement du fichier ZIP")
            exit(1)
    else:
        print(f"Le fichier ZIP existe déjà : {zip_file_path}")

    # Extraction ZIP file
    print("Extraction du fichier ZIP...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_folder)
    print("Extraction terminée.")

    # Verfier if csv exist after extraction
    if os.path.exists(csv_path):
        print(f"Le fichier CSV a été extrait avec succès : {csv_path}")
    else:
        print("Le fichier CSV n'a pas été trouvé après extraction.")

print("Processus terminé.")
