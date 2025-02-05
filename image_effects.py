#image_effects.py
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os


class ImageEffects:
    def __init__(self):
        self.effects = {
            'original': lambda img: img,
            'grayscale': self.grayscale,
            'sepia': self.sepia,
            'pixelate': self.pixelate,
            'blur': self.blur
        }

    def grayscale(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error in grayscale effect: {e}")
            return image

    def sepia(self, image):
        try:
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(image, sepia_filter)
            sepia_img[np.where(sepia_img > 255)] = 255
            return sepia_img.astype(np.uint8)
        except Exception as e:
            print(f"Error in sepia effect: {e}")
            return image

    def pixelate(self, image, pixel_size=10):
        try:
            h, w = image.shape[:2]
            small = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error in pixelate effect: {e}")
            return image

    def blur(self, image):
        try:
            return cv2.GaussianBlur(image, (15, 15), 0)
        except Exception as e:
            print(f"Error in blur effect: {e}")
            return image

    def process_image(self, image_path):
        """Process image with all effects and return base64 encoded results"""
        print(f"Processing image: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read image using PIL first
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array for OpenCV processing
            image = np.array(pil_image)
            # Convert from RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

        results = {}
        for effect_name, effect_func in self.effects.items():
            try:
                # Apply effect
                processed = effect_func(image.copy())
                # Convert back to RGB
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_processed = Image.fromarray(processed_rgb)

                # Save to base64
                buffer = io.BytesIO()
                pil_processed.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()

                results[effect_name] = {
                    'image': img_str,
                    'title': effect_name.capitalize(),
                    'description': f"{effect_name.capitalize()} effect applied to the image"
                }
            except Exception as e:
                print(f"Error processing {effect_name} effect: {str(e)}")
                continue

        return results
 #
