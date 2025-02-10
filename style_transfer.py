# style_transfer.py
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


class StyleTransfer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # Mean and std for image normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        # Define the layer mapping for features
        self.content_layers_map = {'21': 'conv_4'}  # conv4_2
        self.style_layers_map = {
            '0': 'conv_1',  # conv1_1
            '5': 'conv_2',  # conv2_1
            '10': 'conv_3',  # conv3_1
            '19': 'conv_4',  # conv4_1
            '28': 'conv_5'  # conv5_1
        }

        # Content and style layers
        self.content_layers = list(self.content_layers_map.values())
        self.style_layers = list(self.style_layers_map.values())

        # Weights for loss
        self.content_weight = 1
        self.style_weight = 1e6

    def load_image(self, image_path, max_size=512):
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')

            # Resize while maintaining aspect ratio
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)

            # Transform to tensor
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # Move tensor to device and add batch dimension
            image = transform(image).unsqueeze(0).to(self.device)
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")

    def preprocess(self, image):
        """Normalize image"""
        image = (image - self.mean) / self.std
        return image

    def deprocess(self, image):
        """Denormalize image"""
        image = image * self.std + self.mean
        image = image.clamp_(0, 1)
        return image

    def get_features(self, image, model):
        """Extract features from specified layers"""
        features = {}
        x = image

        # Loop through model layers and collect features
        for name, layer in model._modules.items():
            x = layer(x)

            # Check if current layer is a content layer
            if name in self.content_layers_map:
                features[self.content_layers_map[name]] = x

            # Check if current layer is a style layer
            if name in self.style_layers_map:
                features[self.style_layers_map[name]] = x

        return features

    def gram_matrix(self, tensor):
        """Calculate Gram matrix"""
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram.div_(h * w)

    def style_transfer(self, content_path, style_path, num_steps=300):
        """Perform style transfer"""
        try:
            # Load and preprocess images
            content = self.load_image(content_path)
            style = self.load_image(style_path)

            # Preprocess images
            content = self.preprocess(content)
            style = self.preprocess(style)

            # Initialize target image with content image
            target = content.clone().requires_grad_(True)

            # Get features
            with torch.no_grad():
                content_features = self.get_features(content, self.vgg)
                style_features = self.get_features(style, self.vgg)
                style_grams = {layer: self.gram_matrix(style_features[layer])
                               for layer in self.style_layers}

            # Optimizer
            optimizer = torch.optim.Adam([target], lr=0.01)

            # Style transfer loop
            for step in range(num_steps):
                def closure():
                    # Zero gradients
                    optimizer.zero_grad()

                    # Get current target features
                    target_features = self.get_features(target, self.vgg)

                    # Content loss
                    content_loss = 0
                    for layer in self.content_layers:
                        target_feature = target_features[layer]
                        content_feature = content_features[layer].detach()
                        content_loss += F.mse_loss(target_feature, content_feature)

                    # Style loss
                    style_loss = 0
                    for layer in self.style_layers:
                        target_feature = target_features[layer]
                        target_gram = self.gram_matrix(target_feature)
                        style_gram = style_grams[layer].detach()
                        style_loss += F.mse_loss(target_gram, style_gram)

                    # Total loss
                    total_loss = self.content_weight * content_loss + self.style_weight * style_loss

                    # Compute gradients
                    total_loss.backward()

                    return total_loss

                # Update target image
                optimizer.step(closure)

            # Deprocess and return final image
            with torch.no_grad():
                final_img = self.deprocess(target)

            return final_img.cpu().squeeze(0)

        except Exception as e:
            raise Exception(f"Style transfer failed: {str(e)}")