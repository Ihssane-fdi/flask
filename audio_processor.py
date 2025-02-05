# audio_processor.py
from pydub import AudioSegment
import simpleaudio as sa
import os
from datetime import datetime
from werkzeug.utils import secure_filename


class AudioProcessor:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.preview_folder = os.path.join(audio_folder, 'previews')
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(self.preview_folder, exist_ok=True)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3'}

    def load_audio(self, file_path):
        if file_path.lower().endswith('.mp3'):
            return AudioSegment.from_mp3(file_path)
        return AudioSegment.from_wav(file_path)

    def process_audio(self, input_file, speed=1.5, fade_in=1000, fade_out=1000, save_as_preview=True):
        try:
            # Load the audio
            audio = self.load_audio(input_file)

            # Apply speed change
            modified_audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)
            })

            # Apply fades
            modified_audio = modified_audio.fade_in(fade_in).fade_out(fade_out)

            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = os.path.basename(input_file)
            filename_without_ext = os.path.splitext(original_filename)[0]

            if save_as_preview:
                output_filename = f"{filename_without_ext}_preview_{timestamp}.wav"
                output_path = os.path.join(self.preview_folder, secure_filename(output_filename))
            else:
                output_filename = f"{filename_without_ext}_modified_{timestamp}.wav"
                output_path = os.path.join(self.audio_folder, secure_filename(output_filename))

            # Save the modified audio
            modified_audio.export(output_path, format="wav")

            return output_filename, {
                'speed': speed,
                'fade_in': fade_in,
                'fade_out': fade_out
            }

        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None

    def get_audio_files(self):
        """Get list of all audio files with their preview status"""
        if not os.path.exists(self.audio_folder):
            return []

        audio_files = []
        for f in os.listdir(self.audio_folder):
            if f.lower().endswith(('.wav', '.mp3')) and not os.path.isdir(os.path.join(self.audio_folder, f)):
                # Check if there's a preview version
                preview_files = [p for p in os.listdir(self.preview_folder)
                                 if p.startswith(os.path.splitext(f)[0] + '_preview_')]
                preview_file = preview_files[-1] if preview_files else None

                audio_files.append({
                    'filename': f,
                    'preview': preview_file,
                    'is_modified': '_modified_' in f
                })

        return sorted(audio_files, key=lambda x: x['filename'])

    def delete_file(self, filename):
        """Delete a file and its associated previews"""
        # Delete the main file
        filepath = os.path.join(self.audio_folder, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        # Delete any preview files
        filename_without_ext = os.path.splitext(filename)[0]
        for preview in os.listdir(self.preview_folder):
            if preview.startswith(filename_without_ext + '_preview_'):
                os.remove(os.path.join(self.preview_folder, preview))
