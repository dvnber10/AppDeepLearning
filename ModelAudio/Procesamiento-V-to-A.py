import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio_from_mp4(video_path, audio_output_path):
    """
    Extrae la pista de audio de un archivo de video MP4.
    """
    try:
        # Cargar el video
        video_clip = VideoFileClip(video_path)
        
        # Extraer el audio
        audio_clip = video_clip.audivso
        
        # Guardar el audio en formato WAV
        audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le')
        
        print(f"Audio extraído y guardado en: {audio_output_path}")
        
    except Exception as e:
        print(f"Error al extraer el audio: {e}")

# Ejemplo de uso
if __name__ == '__main__':
    # Asegúrate de tener un archivo de video llamado 'tu_video.mp4' en el mismo directorio.
    video_input = "Pruebas/Input/Sway.mp4"
    audio_output = "Pruebas/Output/audio_extraido1.wav"
    
    extract_audio_from_mp4(video_input, audio_output)