#app.py
import uuid
import pygame
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
from pydub import AudioSegment
from drawing_tool import DrawingTool
import os
from datetime import datetime
import base64
from generative import turtle_art_image, pygame_art_image
from visualization import DataVisualization
from image_effects import ImageEffects
from io import BytesIO
from PIL import Image
import traceback
from werkzeug.utils import secure_filename
import imghdr
from audio_processor import AudioProcessor
from generate_descriptions import MLProcessor
import torchvision.transforms as transforms
from style_transfer import StyleTransfer


app = Flask(__name__)


app.config['ARTWORK_FOLDER'] = os.path.join(app.root_path, 'static', 'gallery', 'artworks')
app.config['VISUALIZATION_FOLDER'] = os.path.join(app.root_path, 'static', 'gallery', 'visualizations')

# Ensure upload folders exist
os.makedirs(app.config['ARTWORK_FOLDER'], exist_ok=True)
os.makedirs(app.config['VISUALIZATION_FOLDER'], exist_ok=True)

app.config['DEFAULT_IMAGE'] = os.path.join(app.root_path, 'static', 'default.jpg')  # Add this line

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/draw')
def draw():
    """Launch the Pygame drawing tool"""
    drawing_tool = DrawingTool()
    image_data = drawing_tool.run_tool()

    if image_data is None:
        return render_template('index.html')

    if ',' in image_data:
        image_data = image_data.split(',')[1]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'drawing_{timestamp}.png'
    filepath = os.path.join(app.config['ARTWORK_FOLDER'], filename)

    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(image_data))

    return jsonify({'image': url_for('static', filename='gallery/artworks/' + filename), 'filename': filename})

@app.route('/gallery')
def gallery():
    """Display all saved artwork and visualizations"""
    artwork_files = os.listdir(app.config['ARTWORK_FOLDER'])
    artwork_files = [f for f in artwork_files if f.endswith('.png')]

    visualization_files = os.listdir(app.config['VISUALIZATION_FOLDER'])
    visualization_files = [f for f in visualization_files if f.endswith('.png')]

    return render_template('gallery.html', artworks=artwork_files, visualizations=visualization_files)

@app.route('/generate_turtle_art')
def generate_turtle_art():
    try:
        print("Calling turtle_art_image()...")
        turtle_art_image(app.config['ARTWORK_FOLDER'])
        print("Turtle Art Generated! Redirecting to gallery...")
        return redirect(url_for('gallery'))
    except Exception as e:
        print(f"Error in /generate_turtle_art: {e}")
        return f"Error: {e}", 500

@app.route('/generate_pygame_art')
def generate_pygame_art():
    try:
        print("Calling pygame_art_image()...")
        pygame_art_image(app.config['ARTWORK_FOLDER'])
        print("Pygame Art Generated! Redirecting to gallery...")
        return redirect(url_for('gallery'))
    except Exception as e:
        print(f"Error in /generate_pygame_art: {e}")
        return f"Error: {e}", 500

@app.route('/delete_artwork/<artwork>', methods=['POST'])
def delete_artwork(artwork):
    artwork_path = os.path.join(app.config['ARTWORK_FOLDER'], artwork)
    try:
        if os.path.exists(artwork_path):
            os.remove(artwork_path)
        else:
            print(f"Artwork not found: {artwork}")
    except Exception as e:
        print(f"Error deleting artwork: {e}")
    return redirect(url_for('gallery'))

@app.route('/visualization')
def visualization():
    """Display happiness data visualizations"""
    dv = DataVisualization()
    plots = dv.get_all_plots()
    return render_template('visualization.html', plots=plots)


def validate_image(stream):
    """Validate if the file is a valid image"""
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + format if format != 'jpeg' else '.jpg'


# Add these configurations to your Flask app
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/effects', methods=['GET', 'POST'])
def effects_page():
    """Show the effects page with available images and handle uploads"""
    message = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            message = "No file selected"
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            message = "No file selected"
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            # Add timestamp to filename to make it unique
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            filepath = os.path.join(app.config['ARTWORK_FOLDER'], filename)

            try:
                # Validate and save the image
                file_ext = validate_image(file.stream)
                if not file_ext:
                    message = "Invalid image file"
                    return redirect(request.url)

                file.save(filepath)
                # Redirect to effects page for this image
                return redirect(url_for('image_effects', image_name=filename))
            except Exception as e:
                message = f"Error saving file: {str(e)}"
                return redirect(request.url)
        else:
            message = "Invalid file type. Allowed types: PNG, JPG, JPEG, GIF"
            return redirect(request.url)

    # Get list of available images
    artwork_files = []
    if os.path.exists(app.config['ARTWORK_FOLDER']):
        artwork_files = [f for f in os.listdir(app.config['ARTWORK_FOLDER'])
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Create effects instance to get effect descriptions
    effects_processor = ImageEffects()
    effects_list = {
        name: {'title': name.capitalize(), 'description': f"Apply {name} effect to your image"}
        for name in effects_processor.effects.keys()
    }

    return render_template('effects_select.html',
                           images=artwork_files,
                           effects=effects_list,
                           message=message)

@app.route('/effects/<image_name>')
def image_effects(image_name):
    """Apply effects to a specific image"""
    try:
        image_path = os.path.join(app.config['ARTWORK_FOLDER'], image_name)
        print(f"Processing image: {image_path}")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return "Image not found", 404

        # Verify the image can be opened
        try:
            with Image.open(image_path) as img:
                print(f"Successfully opened image: {image_path}")
                print(f"Image size: {img.size}")
                print(f"Image mode: {img.mode}")
        except Exception as e:
            print(f"Error verifying image: {str(e)}")
            return f"Error verifying image: {str(e)}", 500

        effects = ImageEffects()
        all_effects = effects.process_image(image_path)

        if not all_effects:
            return "Error processing image effects", 500

        return render_template('effects.html',
                               effects=all_effects,
                               original_image=image_name)

    except Exception as e:
        print(f"Error in image_effects: {str(e)}")
        traceback.print_exc()
        return f"Error processing image: {str(e)}", 500


# Add these routes to app.py

app.config['AUDIO_FOLDER'] = os.path.join(app.root_path, 'static', 'audio')
audio_processor = AudioProcessor(app.config['AUDIO_FOLDER'])


@app.route('/audio')
def audio_page():
    audio_files = audio_processor.get_audio_files()
    return render_template('audio.html', audio_files=audio_files)


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not file:
        return redirect(request.url)

    if audio_processor.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)
        file.save(filepath)

        effects = {
            'speed': float(request.form.get('speed', 1.0)),
            'fade_in': int(request.form.get('fade_in', 0)),
            'fade_out': int(request.form.get('fade_out', 0)),
            'volume': float(request.form.get('volume', 0)),
            'reverse': request.form.get('reverse') == 'on',
            'loop': int(request.form.get('loop', 1)),
            'trim_start': int(request.form.get('trim_start', 0)),
            'trim_end': int(request.form.get('trim_end', 0))
        }

        preview_filename, _ = audio_processor.process_audio(
            filepath,
            effects=effects,
            save_as_preview=True
        )

    return redirect(url_for('audio_page'))


@app.route('/layer_audio', methods=['POST'])
def layer_audio():
    file_ids = request.form.getlist('audio_files')
    effects_list = []

    for i in range(len(file_ids)):
        if file_ids[i]:  # Only add effects for selected files
            effects = {
                'volume': float(request.form.get(f'volume_{i}', 0)),
                'speed': float(request.form.get(f'speed_{i}', 1.0)),
                'loop': int(request.form.get(f'loop_{i}', 1))
            }
            effects_list.append(effects)

    output_filename = audio_processor.layer_audio(file_ids, effects_list)
    return redirect(url_for('audio_page'))


@app.route('/save_modified/<filename>')
def save_modified(filename):
    preview_path = os.path.join(app.config['AUDIO_FOLDER'], 'previews', filename)

    if os.path.exists(preview_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"modified_{timestamp}.wav"
        new_path = os.path.join(app.config['AUDIO_FOLDER'], new_filename)

        audio = AudioSegment.from_wav(preview_path)
        audio.export(new_path, format="wav")

    return redirect(url_for('audio_page'))


@app.route('/delete_audio/<path:filename>')
def delete_audio(filename):
    try:
        if filename.startswith('layers/'):
            filepath = os.path.join(app.config['AUDIO_FOLDER'], 'layers', os.path.basename(filename))
        else:
            filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)

        if os.path.exists(filepath):
            os.remove(filepath)

            # Delete associated previews
            filename_without_ext = os.path.splitext(os.path.basename(filename))[0]
            preview_folder = os.path.join(app.config['AUDIO_FOLDER'], 'previews')
            for preview in os.listdir(preview_folder):
                if preview.startswith(filename_without_ext + '_preview_'):
                    os.remove(os.path.join(preview_folder, preview))
    except Exception as e:
        print(f"Error deleting file: {e}")

    return redirect(url_for('audio_page'))

ml_processor = MLProcessor(app)


# Add these routes to app.py
@app.route('/generate_descriptions')
def generate_descriptions_page():
    """Display description generation page"""
    artwork_files = []
    if os.path.exists(app.config['ARTWORK_FOLDER']):
        artwork_files = [f for f in os.listdir(app.config['ARTWORK_FOLDER'])
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return render_template('generate_descriptions.html', images=artwork_files)

@app.route('/generate_description/<image_name>', methods=['POST'])
def generate_description(image_name):
    """Generate a description for an artwork"""
    try:
        image_path = os.path.join(app.config['ARTWORK_FOLDER'], image_name)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404

        description = ml_processor.generate_artwork_description(image_path)
        return jsonify({'description': description})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


app.config.update(dict(
    ARTWORK_FOLDER=os.path.join(app.static_folder, 'gallery', 'artworks'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    ALLOWED_EXTENSIONS={'.png', '.jpg', '.jpeg'}
))


# Ensure upload folder exists
os.makedirs(app.config['ARTWORK_FOLDER'], exist_ok=True)
# Initialize style transfer instance
style_transfer_model = StyleTransfer()


@app.route('/style_transfer')
def style_transfer_page():
    """Display style transfer page"""
    try:
        artwork_files = []
        if os.path.exists(app.config['ARTWORK_FOLDER']):
            artwork_files = [f for f in os.listdir(app.config['ARTWORK_FOLDER'])
                             if os.path.splitext(f.lower())[1] in app.config['ALLOWED_EXTENSIONS']]

        return render_template('style_transfer.html',
                               images=sorted(artwork_files),
                               max_file_size=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/style_transfer', methods=['POST'])
def apply_style_transfer():
    """Apply style transfer to images"""
    try:
        # Validate request
        if 'content_image' not in request.form or 'style_image' not in request.form:
            return jsonify({'error': 'Missing required images'}), 400

        content_image = secure_filename(request.form['content_image'])
        style_image = secure_filename(request.form['style_image'])

        content_path = os.path.join(app.config['ARTWORK_FOLDER'], content_image)
        style_path = os.path.join(app.config['ARTWORK_FOLDER'], style_image)

        # Validate files exist
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content image not found'}), 404
        if not os.path.exists(style_path):
            return jsonify({'error': 'Style image not found'}), 404

        # Create style transfer instance if not exists
        if not hasattr(app, 'style_transfer_model'):
            app.style_transfer_model = StyleTransfer()

        # Apply style transfer
        result_img = app.style_transfer_model.style_transfer(
            content_path,
            style_path,
            num_steps=300  # Reduced steps for faster processing
        )

        # Save result
        output_filename = f"style_transfer_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(app.config['ARTWORK_FOLDER'], output_filename)
        result_img.save(output_path, 'PNG')

        return jsonify({
            'output_image': url_for('static',
                                    filename=f'gallery/artworks/{output_filename}'),
            'filename': output_filename,
            'message': 'Style transfer complete!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)