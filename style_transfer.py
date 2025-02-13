import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # Load pretrained VGG19 model
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Split VGG19 into sections for content and style
        self.conv1_1 = nn.Sequential(*list(vgg19.children())[:2])
        self.conv2_1 = nn.Sequential(*list(vgg19.children())[2:7])
        self.conv3_1 = nn.Sequential(*list(vgg19.children())[7:12])
        self.conv4_1 = nn.Sequential(*list(vgg19.children())[12:21])
        self.conv5_1 = nn.Sequential(*list(vgg19.children())[21:30])

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        x = self.conv1_1(x)
        features.append(x)
        x = self.conv2_1(x)
        features.append(x)
        x = self.conv3_1(x)
        features.append(x)
        x = self.conv4_1(x)
        features.append(x)
        x = self.conv5_1(x)
        features.append(x)
        return features


class StyleTransfer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = StyleTransferModel().to(device).eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Image deprocessing
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1]),
        ])

        # Style weights for different layers
        self.style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        }

        # Content weight
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

            # Transform and add batch dimension
            image = self.transform(image).unsqueeze(0).to(self.device)
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")

    def gram_matrix(self, tensor):
        """Calculate Gram Matrix"""
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def style_transfer(self, content_path, style_path, num_steps=500, content_weight=1, style_weight=1e6):
        """Perform neural style transfer"""
        try:
            # Load images
            content_img = self.load_image(content_path)
            style_img = self.load_image(style_path)

            # Initialize target image
            target = content_img.clone().requires_grad_(True)

            # Compute style features
            style_features = self.model(style_img)
            style_grams = [self.gram_matrix(feat) for feat in style_features]

            # Content features (using conv4_1)
            content_features = self.model(content_img)[3].detach()

            # Optimizer
            optimizer = torch.optim.LBFGS([target])

            # Style transfer loop
            run = [0]
            while run[0] <= num_steps:

                def closure():
                    optimizer.zero_grad()

                    # Get current features
                    target_features = self.model(target)

                    # Content loss
                    content_loss = torch.mean((target_features[3] - content_features) ** 2)

                    # Style loss
                    style_loss = 0
                    for ft, gm, weight in zip(target_features, style_grams, self.style_weights.values()):
                        target_gram = self.gram_matrix(ft)
                        style_loss += weight * torch.mean((target_gram - gm) ** 2)

                    # Total loss
                    total_loss = content_weight * content_loss + style_weight * style_loss

                    # Compute gradients
                    total_loss.backward()

                    # Print progress
                    if run[0] % 50 == 0:
                        print(
                            f'Step {run[0]}: Style Loss: {style_loss.item():.4f} Content Loss: {content_loss.item():.4f}')

                    run[0] += 1
                    return total_loss

                optimizer.step(closure)

            # Denormalize and convert to image
            with torch.no_grad():
                target = self.denormalize(target.squeeze(0).cpu())
                target = torch.clamp(target, 0, 1)

            return transforms.ToPILImage()(target)

        except Exception as e:
            raise Exception(f"Style transfer failed: {str(e)}")