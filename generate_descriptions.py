import torch
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import numpy as np
import os


class MLProcessor:
    def __init__(self, app):
        self.app = app
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize GPT-2 for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.text_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

        # Initialize VGG19 for style transfer
        self.style_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()

        # Style layers we're interested in
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize for faster processing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Create output directories
        self.output_dir = os.path.join(app.root_path, 'static', 'gallery', 'artworks')
        self.ml_output_dir = os.path.join(app.root_path, 'static', 'gallery', 'ml_output')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ml_output_dir, exist_ok=True)

    def load_image(self, path):
        """Load and preprocess image"""
        image = Image.open(path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def generate_artwork_description(self, image_path):
        """Generate a description based on image analysis"""
        try:
            image = Image.open(image_path).convert('RGB')

            # Reduce resolution for faster processing
            image = image.resize((256, 256))

            img_array = np.array(image)

            # Analyze image properties
            avg_color = np.mean(img_array, axis=(0, 1))
            brightness = np.mean(avg_color)
            width, height = image.size
            aspect = "portrait" if height > width else "landscape"

            # Create a more detailed prompt
            prompt = (
                f"This {aspect} artwork is a {'bright' if brightness > 128 else 'dark'} piece. "
                f"The dominant colors are RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])}). "
                f"The artwork likely depicts"
            )

            # Generate description using GPT-2
            inputs = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

            outputs = self.text_model.generate(
                inputs,
                max_length=100,  # Reduced max length for faster processing
                do_sample=True,
                temperature=0.6,  # Lower temperature for more structured output
                top_k=40,  # Lower value for faster output
                top_p=0.8,  # Smaller top-p for efficiency
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return description

        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return f"Error analyzing image: {str(e)}"
