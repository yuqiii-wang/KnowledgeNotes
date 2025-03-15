from pydub import AudioSegment

def convert_wav_to_mp3(input_wav, output_mp3):
    # Load the WAV file
    audio = AudioSegment.from_wav(input_wav)
    
    # Export as MP3 without manually specifying bitrate (defaults to source quality)
    audio.export(output_mp3, format="mp3")
    
    print(f"Converted {input_wav} to {output_mp3} with auto bitrate")

# Example usage
convert_wav_to_mp3("hello_world_good_stretch.wav", "hello_world_good_stretch.mp3")