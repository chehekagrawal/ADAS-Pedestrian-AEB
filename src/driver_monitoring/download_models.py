import urllib.request
import bz2
import os

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_DIR = "models"
BZ2_PATH = os.path.join(OUTPUT_DIR, "shape_predictor_68_face_landmarks.dat.bz2")
DAT_PATH = os.path.join(OUTPUT_DIR, "shape_predictor_68_face_landmarks.dat")

def download_and_extract():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if os.path.exists(DAT_PATH):
        print(f"Model already exists at {DAT_PATH}")
        return
        
    print(f"Downloading dlib shape predictor from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, BZ2_PATH)
    print("Download complete.")
    
    print("Extracting .bz2 file...")
    with bz2.BZ2File(BZ2_PATH, 'rb') as source, open(DAT_PATH, 'wb') as dest:
        dest.write(source.read())
        
    print(f"Extraction complete! Model saved to {DAT_PATH}")
    os.remove(BZ2_PATH)

if __name__ == "__main__":
    download_and_extract()
