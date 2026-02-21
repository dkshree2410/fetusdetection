import os
import numpy as np
import cv2


def create_severity_sample_data():
    """Create sample severity data for CHD severity classification"""

    # Create directories
    for severity in ['mild', 'moderate', 'severe']:
        os.makedirs(f'data/severity_data/{severity}', exist_ok=True)

    print("Creating sample CHD severity data...")

    # Mild cases (small abnormalities - mostly normal with minor issues)
    print("Creating MILD cases...")
    for i in range(30):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture (ultrasound-like)
        noise = np.random.normal(0, 12, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Mostly normal heart structure with minor asymmetry
        cv2.ellipse(img, (150, 150), (55, 45), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (250, 150), (52, 45), 0, 0, 360, (180, 180, 180), -1)  # Slightly smaller right chamber

        # Small abnormality (minor septal defect)
        cv2.ellipse(img, (200, 150), (8, 5), 0, 0, 360, (200, 150, 150), -1)

        # Add some normal structures
        cv2.ellipse(img, (200, 100), (12, 8), 0, 0, 360, (160, 160, 160), -1)  # Normal vessel

        # Text labels
        cv2.putText(img, "MILD CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        cv2.putText(img, "Minor Abnormality", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/mild/mild_{i + 1:03d}.png', img)
        if (i + 1) % 10 == 0:
            print(f"  Created mild sample {i + 1}/30")

    # Moderate cases (more noticeable abnormalities)
    print("Creating MODERATE cases...")
    for i in range(30):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture
        noise = np.random.normal(0, 12, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Clear asymmetry between chambers
        cv2.ellipse(img, (140, 150), (65, 50), 0, 0, 360, (180, 180, 180), -1)  # Larger left chamber
        cv2.ellipse(img, (260, 150), (45, 50), 0, 0, 360, (180, 180, 180), -1)  # Smaller right chamber

        # Multiple abnormalities
        cv2.ellipse(img, (200, 80), (20, 12), 0, 0, 360, (200, 120, 120), -1)  # Upper abnormality
        cv2.ellipse(img, (180, 200), (15, 10), 0, 0, 360, (200, 120, 120), -1)  # Lower abnormality

        # Irregular vessel structure
        cv2.ellipse(img, (200, 100), (18, 25), 30, 0, 360, (150, 150, 150), -1)  # Irregular vessel

        # Text labels
        cv2.putText(img, "MODERATE CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 50), 2)
        cv2.putText(img, "Clear Abnormalities", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 50), 1)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/moderate/moderate_{i + 1:03d}.png', img)
        if (i + 1) % 10 == 0:
            print(f"  Created moderate sample {i + 1}/30")

    # Severe cases (major structural abnormalities)
    print("Creating SEVERE cases...")
    for i in range(30):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Add background texture
        noise = np.random.normal(0, 12, (300, 400, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        # Severe asymmetry and structural issues
        cv2.ellipse(img, (130, 150), (75, 55), 0, 0, 360, (180, 180, 180), -1)  # Very large left chamber
        cv2.ellipse(img, (270, 150), (35, 55), 0, 0, 360, (180, 180, 180), -1)  # Very small right chamber

        # Multiple major abnormalities
        cv2.ellipse(img, (200, 70), (25, 18), 0, 0, 360, (200, 100, 100), -1)  # Large upper defect
        cv2.ellipse(img, (180, 200), (22, 16), 0, 0, 360, (200, 100, 100), -1)  # Large lower defect
        cv2.ellipse(img, (220, 120), (15, 12), 0, 0, 360, (200, 100, 100), -1)  # Additional defect

        # Distorted vessel structures
        points = np.array([[180, 50], [220, 60], [230, 90], [210, 110], [170, 100]], np.int32)
        cv2.fillPoly(img, [points], (140, 140, 140))  # Irregular vessel

        # Text labels
        cv2.putText(img, "SEVERE CHD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(img, "Major Structural Issues", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/severity_data/severe/severe_{i + 1:03d}.png', img)
        if (i + 1) % 10 == 0:
            print(f"  Created severe sample {i + 1}/30")

    print("\nSample severity data created successfully!")
    print("Mild cases: 30 images")
    print("Moderate cases: 30 images")
    print("Severe cases: 30 images")
    print("Total: 90 severity classification images")


def verify_severity_data():
    """Verify that the severity data was created correctly"""
    print("\nVerifying severity data...")

    base_dir = 'data/severity_data'
    severity_levels = ['mild', 'moderate', 'severe']

    for severity in severity_levels:
        severity_dir = os.path.join(base_dir, severity)
        if os.path.exists(severity_dir):
            files = [f for f in os.listdir(severity_dir) if f.endswith('.png')]
            print(f"  {severity}: {len(files)} images")

            # Show sample files
            if files:
                print(f"    Sample: {files[0]}")
        else:
            print(f"  {severity}: Directory not found!")

    total_files = sum([len([f for f in os.listdir(os.path.join(base_dir, s)) if f.endswith('.png')])
                       for s in severity_levels if os.path.exists(os.path.join(base_dir, s))])
    print(f"\nTotal severity images: {total_files}")


def create_sample_description():
    """Create a description file explaining the severity levels"""
    description = """
CHD Severity Levels Description:

MILD (30 samples):
- Minor structural abnormalities
- Slight asymmetry between heart chambers
- Small septal defects
- Overall heart function largely preserved

MODERATE (30 samples):
- Clear structural abnormalities
- Noticeable chamber asymmetry
- Multiple minor defects
- Some impact on heart function

SEVERE (30 samples):
- Major structural abnormalities
- Significant chamber asymmetry
- Multiple large defects
- Severe impact on heart function
- Distorted vessel structures
"""

    with open('data/severity_data/severity_levels_description.txt', 'w') as f:
        f.write(description)
    print("Severity levels description saved!")


if __name__ == '__main__':
    print("=== Creating Sample CHD Severity Data ===")
    create_severity_sample_data()
    verify_severity_data()
    create_sample_description()
    print("\nSeverity data creation completed!")
    print("Data saved in: data/severity_data/")