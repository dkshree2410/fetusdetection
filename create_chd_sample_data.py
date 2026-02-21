import os
import numpy as np
import cv2
from PIL import Image


def create_chd_sample_data():
    """Create sample CHD classification data"""

    # Create directories
    os.makedirs('data/chd_classification/chd_positive', exist_ok=True)
    os.makedirs('data/chd_classification/chd_negative', exist_ok=True)

    print("Creating sample CHD classification data...")

    # Create CHD Positive samples (abnormal heart structures)
    for i in range(50):
        # Create ultrasound-like background
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add abnormal heart patterns (irregular shapes, asymmetrical)
        # Left side larger than right (simulating CHD)
        cv2.ellipse(img, (150, 150), (80, 60), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (250, 150), (40, 60), 0, 0, 360, (180, 180, 180), -1)  # Smaller right side

        # Add some irregular patterns
        for j in range(5):
            x = np.random.randint(100, 300)
            y = np.random.randint(100, 200)
            w = np.random.randint(10, 30)
            h = np.random.randint(10, 30)
            cv2.ellipse(img, (x, y), (w, h), np.random.randint(0, 180), 0, 360,
                        (np.random.randint(150, 200),) * 3, -1)

        # Add text indicating CHD
        cv2.putText(img, "CHD POSITIVE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(img, "Abnormal Structure", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Save as CHD positive
        filename = f"data/chd_classification/chd_positive/chd_positive_{i + 1:03d}.png"
        cv2.imwrite(filename, img)
        print(f"Created CHD positive sample: {filename}")

    # Create CHD Negative samples (normal heart structures)
    for i in range(50):
        # Create ultrasound-like background
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add normal heart patterns (symmetrical, regular shapes)
        # Symmetrical chambers
        cv2.ellipse(img, (150, 150), (60, 50), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (250, 150), (60, 50), 0, 0, 360, (180, 180, 180), -1)

        # Add regular patterns
        for j in range(3):
            x = np.random.randint(120, 280)
            y = np.random.randint(120, 180)
            radius = np.random.randint(15, 25)
            cv2.circle(img, (x, y), radius, (np.random.randint(160, 200),) * 3, -1)

        # Add text indicating normal
        cv2.putText(img, "NORMAL HEART", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(img, "Normal Structure", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Save as CHD negative
        filename = f"data/chd_classification/chd_negative/chd_negative_{i + 1:03d}.png"
        cv2.imwrite(filename, img)
        print(f"Created CHD negative sample: {filename}")

    print(f"\nCreated {50} CHD positive and {50} CHD negative samples")
    print("Sample CHD data ready for training!")


if __name__ == '__main__':
    create_chd_sample_data()