import kagglehub

# Download latest version
path = kagglehub.dataset_download("warcoder/earthquake-dataset")

print("Path to dataset files:", path)