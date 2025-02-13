from pydub import AudioSegment
import os
from datetime import datetime
from werkzeug.utils import secure_filename


class AudioProcessor:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.preview_folder = os.path.join(audio_folder, 'previews')
        self.layers_folder = os.path.join(audio_folder, 'layers')
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(self.preview_folder, exist_ok=True)
        os.makedirs(self.layers_folder, exist_ok=True)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3'}

    def load_audio(self, file_path):
        if file_path.lower().endswith('.mp3'):
            return AudioSegment.from_mp3(file_path)
        return AudioSegment.from_wav(file_path)

    def process_audio(self, input_file, effects=None, save_as_preview=True):
        try:
            audio = self.load_audio(input_file)
            modified_audio = self.apply_effects(audio, effects) if effects else audio

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = os.path.basename(input_file)
            filename_without_ext = os.path.splitext(original_filename)[0]

            output_filename = f"{filename_without_ext}_preview_{timestamp}.wav"
            output_path = os.path.join(self.preview_folder, secure_filename(output_filename))
            modified_audio.export(output_path, format="wav")

            return output_filename, effects

        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None

    def apply_effects(self, audio, effects):
        if not effects:
            return audio

        modified = audio

        if 'trim_start' in effects and effects['trim_start'] > 0:
            start_ms = effects['trim_start']
            if start_ms < len(modified):
                modified = modified[start_ms:]

        if 'trim_end' in effects and effects['trim_end'] > 0:
            end_ms = effects['trim_end']
            if end_ms < len(modified):
                modified = modified[:-end_ms]

        if 'speed' in effects and effects['speed'] != 1.0:
            modified = modified._spawn(modified.raw_data, overrides={
                "frame_rate": int(modified.frame_rate * effects['speed'])
            })

        if 'fade_in' in effects and effects['fade_in'] > 0:
            modified = modified.fade_in(effects['fade_in'])

        if 'fade_out' in effects and effects['fade_out'] > 0:
            modified = modified.fade_out(effects['fade_out'])

        if 'volume' in effects and effects['volume'] != 0:
            modified += effects['volume']

        if 'reverse' in effects and effects['reverse']:
            modified = modified.reverse()

        if 'loop' in effects and effects['loop'] > 1:
            modified = modified * effects['loop']

        return modified

    def layer_audio(self, file_ids, effects_list=None):
        try:
            if not file_ids:
                return None

            final_audio = None
            for i, file_id in enumerate(file_ids):
                if not file_id:  # Skip empty selections
                    continue

                filepath = os.path.join(self.audio_folder, file_id)
                if not os.path.exists(filepath):
                    continue

                audio = self.load_audio(filepath)

                if effects_list and i < len(effects_list):
                    audio = self.apply_effects(audio, effects_list[i])

                if final_audio is None:
                    final_audio = audio
                else:
                    final_audio = final_audio.overlay(audio)

            if final_audio:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"layered_{timestamp}.wav"
                output_path = os.path.join(self.layers_folder, output_filename)
                final_audio.export(output_path, format="wav")
                return output_filename

        except Exception as e:
            print(f"Error layering audio: {e}")
            return None

    def get_audio_files(self):
        audio_files = []

        # Get regular audio files
        for f in os.listdir(self.audio_folder):
            if self.allowed_file(f) and not os.path.isdir(os.path.join(self.audio_folder, f)):
                preview_files = [p for p in os.listdir(self.preview_folder)
                                 if p.startswith(os.path.splitext(f)[0] + '_preview_')]
                preview_file = preview_files[-1] if preview_files else None

                audio_files.append({
                    'filename': f,
                    'preview': preview_file,
                    'is_modified': False,
                    'is_layer': False
                })

        # Get layered files
        for f in os.listdir(self.layers_folder):
            if self.allowed_file(f):
                audio_files.append({
                    'filename': os.path.join('layers', f),
                    'preview': None,
                    'is_modified': True,
                    'is_layer': True
                })

        return sorted(audio_files, key=lambda x: x['filename'])

    def delete_file(self, filename):
        if filename.startswith('layers/'):
            filepath = os.path.join(self.audio_folder, filename)
        else:
            filepath = os.path.join(self.audio_folder, filename)

        if os.path.exists(filepath):
            os.remove(filepath)

        # Delete previews
        filename_without_ext = os.path.splitext(os.path.basename(filename))[0]
        for preview in os.listdir(self.preview_folder):
            if preview.startswith(filename_without_ext + '_preview_'):
                os.remove(os.path.join(self.preview_folder, preview))