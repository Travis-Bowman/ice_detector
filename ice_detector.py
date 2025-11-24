import tensorflow as tf
import numpy as np
from PIL import Image
from gpiozero import LED   
import time 

MODEL_PATH = "/home/dev/Documents/python_workzone/ice_detector/models/iceNoice.keras"
IMAGE_PATH = "/home/dev/Documents/python_workzone/ice_detector/test_images/no_ice.JPG"
IMAGE_SIZE = (180, 180)
ICE_THRESHOLD = 0.5   # cutoff for ICE vs NO ICE
output_signal = LED(17)  # GPIO pin 17 for output indication

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (H,W,C) → (1,H,W,C)

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print(f"Loading test image: {IMAGE_PATH}")
    x = load_image(IMAGE_PATH)

    preds = model.predict(x)
    prob_ice = float(preds[0][0])  # sigmoid output

    label = "ICE" if prob_ice <= ICE_THRESHOLD else "NO ICE"
    print(f"Ice probability: {prob_ice:.4f} → {label}")
    
    if label == "ICE":
        for _ in range(5):
            output_signal.on()
            time.sleep(0.5)
            output_signal.off()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
