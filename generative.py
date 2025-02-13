from PIL import Image, ImageDraw
import random
import os
from datetime import datetime


def turtle_art_image(artwork_folder):
    """Generate a turtle-style generative art image and save it."""
    try:
        print("Generating Turtle Art...")
        width, height = 800, 600
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        shapes = ['circle', 'square', 'triangle']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'white']

        for _ in range(50):
            shape = random.choice(shapes)
            color = random.choice(colors)
            x, y = random.randint(100, width - 100), random.randint(100, height - 100)
            size = random.randint(10, 50)

            if shape == 'circle':
                draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
            elif shape == 'square':
                draw.rectangle([x - size, y - size, x + size, y + size], fill=color)
            elif shape == 'triangle':
                draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)], fill=color)

        # Ensure the artwork folder exists
        os.makedirs(artwork_folder, exist_ok=True)

        # Save the generated image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'turtle_art_{timestamp}.png'
        file_path = os.path.join(artwork_folder, filename)

        print(f"Saving Turtle Art to: {file_path}")
        img.save(file_path)
        print("Turtle Art Generation Complete!")

    except Exception as e:
        print(f"Error in turtle_art_image: {e}")


def pygame_art_image(artwork_folder):
    try:
        print("Generating Pygame Art...")
        width, height = 800, 600
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)

        shapes = ['circle', 'square']
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(7)]

        for _ in range(50):
            shape = random.choice(shapes)
            color = random.choice(colors)
            x, y = random.randint(100, width - 100), random.randint(100, height - 100)
            size = random.randint(10, 50)

            if shape == 'circle':
                draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
            elif shape == 'square':
                draw.rectangle([x - size, y - size, x + size, y + size], fill=color)


        os.makedirs(artwork_folder, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'pygame_art_{timestamp}.png'
        file_path = os.path.join(artwork_folder, filename)

        print(f"Saving Pygame Art to: {file_path}")
        img.save(file_path)
        print("Pygame Art Generation Complete!")

    except Exception as e:
        print(f"Error in pygame_art_image: {e}")
