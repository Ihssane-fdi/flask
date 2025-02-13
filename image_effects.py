import cv2
import numpy as np
from PIL import Image
import io
import base64
import os


class ImageEffects:
    def __init__(self):
        self.effects = {
            'original': {
                'func': self.compress_image,  # Add compression to original
                'description': 'Original image without any effects'
            },
            'grayscale': {
                'func': self.grayscale,
                'description': 'Convert image to black and white'
            },
            'sepia': {
                'func': self.sepia,
                'description': 'Apply a vintage sepia tone'
            },
            'pixelate': {
                'func': self.pixelate,
                'description': 'Create a pixelated mosaic effect'
            },
            'blur': {
                'func': self.blur,
                'description': 'Apply Gaussian blur effect'
            }
        }
        self.max_dimension = 1200  # Maximum dimension for any image
        self.jpeg_quality = 85     # JPEG quality for compression

    def compress_image(self, image):
        """Compress image to reduce file size"""
        try:
            # Convert to RGB if necessary
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Resize if image is too large
            height, width = image.shape[:2]
            if max(height, width) > self.max_dimension:
                scale = self.max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            return image
        except Exception as e:
            print(f"Error in compress_image: {e}")
            return image

    def process_image(self, image_path, selected_effects=None):
        """Process image with selected effects and return base64 encoded results"""
        print(f"Processing image: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read and compress image initially
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Resize if necessary
            if max(pil_image.size) > self.max_dimension:
                pil_image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

            # Convert to numpy array for OpenCV
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

        results = {}
        if not selected_effects:
            selected_effects = ['original']

        for effect_name in selected_effects:
            if effect_name in self.effects:
                try:
                    # Apply effect
                    processed = self.effects[effect_name]['func'](image.copy())
                    # Always compress after effect
                    processed = self.compress_image(processed)
                    # Convert back to RGB
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    pil_processed = Image.fromarray(processed_rgb)

                    # Save to base64 with compression
                    buffer = io.BytesIO()
                    pil_processed.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
                    img_str = base64.b64encode(buffer.getvalue()).decode()

                    results[effect_name] = {
                        'image': img_str,
                        'title': effect_name.capitalize(),
                        'description': self.effects[effect_name]['description']
                    }
                except Exception as e:
                    print(f"Error processing {effect_name} effect: {str(e)}")
                    continue

        return results

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

    def save_processed_image(self, base64_image, original_filename, effect_name):
        """Save a processed image from base64 string to the gallery"""
        try:
            # Decode base64 string
            image_data = base64.b64decode(base64_image)

            # Create image from binary data
            image = Image.open(io.BytesIO(image_data))

            # Compress if necessary
            if max(image.size) > self.max_dimension:
                image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

            # Generate new filename with effect name
            filename_without_ext = os.path.splitext(original_filename)[0]
            new_filename = f"{filename_without_ext}_{effect_name}.jpg"  # Changed to jpg

            return new_filename, image
        except Exception as e:
            print(f"Error saving processed image: {e}")
            return None, None
