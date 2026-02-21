import os
import numpy as np
import cv2

def create_severity_sample_data():
    """Create sample severity data"""

    # Create directories
    for severity in ['mild', 'moderate', 'severe']:
        os.makedirs(f'data/severity_data/{severity}', exist_ok=True)

    print("Creating sample severity data...")

    # Mild cases (small abnormalities)
    for i in range(20):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture
        noise = np.random.normal(0, 10, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Mostly normal with small irregularities
        cv2.ellipse(img, (150, 150), (55, 45), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (250, 150), (55, 45), 0, 0, 360, (180, 180, 180), -1)

        # Small abnormality
        cv2.ellipse(img, (200, 100), (15, 10), 0, 0, 360, (200, 150, 150), -1)

        cv2.putText(img, "MILD CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/mild/mild_{i + 1:03d}.png', img)

    # Moderate cases (more noticeable abnormalities)
    for i in range(20):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture
        noise = np.random.normal(0, 10, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Asymmetrical chambers
        cv2.ellipse(img, (140, 150), (65, 50), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (260, 150), (45, 50), 0, 0, 360, (180, 180, 180), -1)

        # Additional abnormalities
        cv2.ellipse(img, (200, 80), (25, 15), 0, 0, 360, (200, 120, 120), -1)

        cv2.putText(img, "MODERATE CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 50), 2)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/moderate/moderate_{i + 1:03d}.png', img)

    # Severe cases (major abnormalities)
    for i in range(20):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture
        noise = np.random.normal(0, 10, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Very asymmetrical
        cv2.ellipse(img, (130, 150), (75, 55), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (270, 150), (35, 55), 0, 0, 360, (180, 180, 180), -1)

        # Multiple abnormalities
        cv2.ellipse(img, (200, 70), (30, 20), 0, 0, 360, (200, 100, 100), -1)
        cv2.ellipse(img, (180, 200), (20, 25), 0, 0, 360, (200, 100, 100), -1)

        cv2.putText(img, "SEVERE CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/severe/severe_{i + 1:03d}.png', img)

    print("Sample severity data created!")
    print("Mild cases: 20")
    print("Moderate cases: 20")
    print("Severe cases: 20")


if __name__ == '__main__':
    print("=== Creating Sample Severity Data ===")
    create_severity_sample_data()
    print("Severity data creation completed!")