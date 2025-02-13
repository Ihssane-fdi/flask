import pygame
import random
import base64
import io
from PIL import Image


class Shape:
    def __init__(self, x, y, shape_type, color, size):
        self.x = x
        self.y = y
        self.shape_type = shape_type
        self.color = color
        self.size = size

    def draw(self, screen):
        if self.shape_type == "circle":
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)
        elif self.shape_type == "square":
            pygame.draw.rect(screen, self.color,
                             (self.x - self.size // 2, self.y - self.size // 2, self.size, self.size))
        elif self.shape_type == "triangle":
            points = [(self.x, self.y - self.size), (self.x - self.size, self.y + self.size),
                      (self.x + self.size, self.y + self.size)]
            pygame.draw.polygon(screen, self.color, points)


class DrawingTool:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Interactive Drawing Tool")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128)]

        self.shapes = []
        self.selected_shape = "circle"
        self.selected_color = random.choice(self.COLORS)
        self.selected_size = 30
        self.holding_shape = None

        self.font = pygame.font.Font(None, 30)

    def draw_menu(self):
        pygame.draw.rect(self.screen, (200, 200, 200), (0, 0, self.WIDTH, 150))
        self.screen.blit(self.font.render("Select Shape:", True, self.BLACK), (10, 10))
        self.screen.blit(self.font.render(f"Select Size (Up & Down): {self.selected_size}", True, self.BLACK), (10, 40))

        # Draw shape selection menu
        shape_positions = [(100, 30), (200, 30), (300, 30)]
        pygame.draw.circle(self.screen, self.BLACK, shape_positions[0], 15)
        pygame.draw.rect(self.screen, self.BLACK, (shape_positions[1][0] - 15, shape_positions[1][1] - 15, 30, 30))
        pygame.draw.polygon(self.screen, self.BLACK, [(shape_positions[2][0], shape_positions[2][1] - 15),
                                                      (shape_positions[2][0] - 15, shape_positions[2][1] + 15),
                                                      (shape_positions[2][0] + 15, shape_positions[2][1] + 15)])

        # Draw color palette
        for i, color in enumerate(self.COLORS):
            pygame.draw.rect(self.screen, color, (450 + i * 40, 150, 30, 30))
            if color == self.selected_color:
                pygame.draw.rect(self.screen, (0, 0, 0), (450 + i * 40, 150, 30, 30), 3)

        # Undo and Clear indicators
        self.screen.blit(self.font.render("Undo (U)", True, self.BLACK), (10, 70))
        self.screen.blit(self.font.render("Clear (C)", True, self.BLACK), (10, 100))


    def get_selected_shape(self, x, y):
        if 85 <= x <= 115 and 15 <= y <= 45:
            return "circle"
        elif 185 <= x <= 215 and 15 <= y <= 45:
            return "square"
        elif 285 <= x <= 315 and 15 <= y <= 45:
            return "triangle"
        return None

    def get_selected_color(self, x, y):
        for i, color in enumerate(self.COLORS):
            if 450 + i * 40 <= x <= 480 + i * 40 and 150 <= y <= 180:
                return color
        return None

    def save_drawing(self):
        """Save the current drawing as a base64 encoded PNG"""
        string_image = pygame.image.tostring(self.screen, 'RGB')
        temp_surface = Image.frombytes('RGB', (self.WIDTH, self.HEIGHT), string_image)

        # Save to bytes buffer
        buffer = io.BytesIO()
        temp_surface.save(buffer, format='PNG')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64


    def run_tool(self):
        running = True
        while running:
            self.screen.fill(self.WHITE)
            self.draw_menu()

            for shape in self.shapes:
                shape.draw(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False  # Exit the Pygame loop
                    pygame.quit()  # Clean up Pygame resources
                    return None  # Return None to indicate no image was saved

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if y < 50:  # Clicked on the shape menu
                        new_shape = self.get_selected_shape(x, y)
                        if new_shape:
                            self.selected_shape = new_shape
                    elif 150 <= y <= 180:  # Clicked on the color menu
                        new_color = self.get_selected_color(x, y)
                        if new_color:
                            self.selected_color = new_color
                    elif 50 <= x <= 150 and 400 <= y <= 450:  # Clicked on "Save" button
                        image_data = self.save_drawing()  # Save the drawing
                        pygame.quit()  # Clean up Pygame resources
                        return image_data  # Return the image data
                    else:
                        if self.selected_shape:
                            self.holding_shape = Shape(x, y, self.selected_shape, self.selected_color,
                                                       self.selected_size)

                elif event.type == pygame.MOUSEBUTTONUP and self.holding_shape:
                    self.shapes.append(self.holding_shape)
                    self.holding_shape = None

                elif event.type == pygame.MOUSEMOTION and self.holding_shape:
                    self.holding_shape.x, self.holding_shape.y = pygame.mouse.get_pos()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_size += 5
                    elif event.key == pygame.K_DOWN:
                        self.selected_size = max(5, self.selected_size - 5)
                    elif event.key == pygame.K_u and self.shapes:
                        self.shapes.pop()
                    elif event.key == pygame.K_c:
                        self.shapes.clear()

            # Draw save button
            pygame.draw.rect(self.screen, (0, 255, 0), (50, 400, 100, 50))
            font = pygame.font.Font(None, 36)
            text = font.render("Save", True, (0, 0, 0))
            self.screen.blit(text, (75, 415))

            if self.holding_shape:
                self.holding_shape.draw(self.screen)

            pygame.display.flip()

        pygame.quit()  # Clean up Pygame resources
        return None  # Return None if the window was closed without saving
