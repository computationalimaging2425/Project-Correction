# Jupyter Notebook: Image Downsampling from 512×512 to 128×128

# Cell 1: Imports
from PIL import Image
import os
from tqdm import tqdm

# Cell 2: Function Definition
# Cell 2: Function Definition (recursive)
def downsample_images(input_dir, output_dir, target_size=(128, 128)):
    """
    Downsamples all images in input_dir (e sue sottocartelle) a target_size
    e ricrea la struttura di directory corrispondente in output_dir.
    """
    # Crea la cartella di base di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Cammina tutta la gerarchia di input_dir
    for root, dirs, files in os.walk(input_dir):
        # Calcola percorso relativo rispetto a input_dir
        rel_path = os.path.relpath(root, input_dir)
        # Costruisci la cartella di destinazione corrispondente
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Elabora tutti i file in questa cartella
        for filename in tqdm(files, desc=f"Processing {rel_path}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(target_dir, filename)
                with Image.open(input_path) as img:
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    img_resized.save(output_path)
    
    print(f"All images have been downsampled and saved to: {output_dir}")


# Cell 3: Example Usage
# Replace the following paths with your own directories, then uncomment to run:
input_dir = "test"
output_dir = "128x128_images"
output_dir = os.path.join(output_dir, input_dir)
downsample_images(input_dir, output_dir)
