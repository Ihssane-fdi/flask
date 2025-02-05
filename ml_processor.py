# ml_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import numpy as np
import os
from datetime import datetime


class MLProcessor:
    def __init__(self, app):
        self.app = app
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize GPT-2 for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.text_model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Initialize VGG19 for style transfer
        self.style_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()

        # Style layers we're interested in
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(512),
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

    def get_features(self, x):
        """Extract features from image using VGG19"""
        features = {}
        for name, layer in self.style_model._modules.items():
            x = layer(x)
            if name in self.style_layers or name in self.content_layers:
                features[name] = x
        return features

    def gram_matrix(self, tensor):
        """Calculate Gram Matrix"""
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        return torch.mm(tensor, tensor.t()) / (b * c * h * w)

    def apply_style_transfer(self, content_path, style_path, num_steps=300):
        """Apply style transfer with improved optimization"""
        print(f"Starting style transfer: content={content_path}, style={style_path}")

        # Load and preprocess images
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)

        # Initialize target image as a clone of the content image
        target = content_img.clone().requires_grad_(True)

        # Extract features from content and style images
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)

        # Compute Gram matrices for style features
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers
        }

        # Use LBFGS optimizer for better convergence
        optimizer = torch.optim.LBFGS([target], lr=1, max_iter=20)

        # Define weights for content and style losses
        style_weight = 1e6
        content_weight = 1

        step = [0]
        while step[0] < num_steps:
            def closure():
                optimizer.zero_grad()

                # Extract features from the target image
                target_features = self.get_features(target)

                # Compute content loss
                content_loss = 0
                for layer in self.content_layers:
                    content_loss += F.mse_loss(target_features[layer], content_features[layer])

                # Compute style loss
                style_loss = 0
                for layer in self.style_layers:
                    target_gram = self.gram_matrix(target_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += F.mse_loss(target_gram, style_gram)

                # Total loss
                total_loss = content_weight * content_loss + style_weight * style_loss

                # Backward pass with graph retention
                total_loss.backward(retain_graph=True)  # Retain the graph for further backward passes

                step[0] += 1
                if step[0] % 50 == 0:
                    print(
                        f'Step {step[0]}/{num_steps}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}')

                return total_loss

            optimizer.step(closure)

        # Post-process the output image
        with torch.no_grad():
            target_image = target.cpu().squeeze(0)
            target_image = target_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            target_image = target_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            target_image = torch.clamp(target_image, 0, 1)

        # Convert to PIL image and save
        to_pil = transforms.ToPILImage()
        result = to_pil(target_image)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'style_transfer_{timestamp}.png'
        output_path = os.path.join(self.output_dir, filename)
        result.save(output_path, 'PNG', quality=95)

        print(f"Style transfer complete: saved to {output_path}")
        return filename

    def generate_artwork_description(self, image_path):
        """Generate a description based on image analysis"""
        try:
            image = Image.open(image_path).convert('RGB')
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
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(inputs).to(self.device)

            outputs = self.text_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=150,  # Increased max length
                do_sample=True,
                temperature=0.7,  # Adjusted temperature for more coherent output
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return description

        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return f"Error analyzing image: {str(e)}"
