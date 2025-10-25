import requests
import pandas as pd
import time
import random
import re
from bs4 import BeautifulSoup
import json
import os
from deep_translator import GoogleTranslator
from tqdm import tqdm

class LyricsDatasetCreator:
    def __init__(self):
        self.dataset = []
        self.translator = GoogleTranslator(source='en', target='es')
        
    def get_genius_lyrics(self, song_title, artist_name):
        """Obtiene letras de Genius API"""
        try:
            # Headers para simular navegador
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Buscar canción
            search_url = f"https://genius.com/api/search/multi?q={artist_name} {song_title}"
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Encontrar resultado de canción
                for section in data['response']['sections']:
                    if section['type'] == 'song':
                        for hit in section['hits']:
                            if hit['type'] == 'song':
                                song_url = hit['result']['url']
                                
                                # Obtener letra de la página
                                lyric_response = requests.get(song_url, headers=headers, timeout=10)
                                if lyric_response.status_code == 200:
                                    soup = BeautifulSoup(lyric_response.text, 'html.parser')
                                    
                                    # Encontrar div con letras
                                    lyrics_div = soup.find('div', {'data-lyrics-container': 'true'})
                                    if lyrics_div:
                                        # Limpiar letras
                                        lyrics = lyrics_div.get_text(separator='\n')
                                        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remover [Corus], [Verse], etc.
                                        lyrics = re.sub(r'\n+', '\n', lyrics).strip()
                                        return lyrics
            return None
            
        except Exception as e:
            print(f"Error obteniendo letras: {e}")
            return None
    
    def get_musixmatch_lyrics(self, song_title, artist_name):
        """Intenta obtener letras de Musixmatch (método alternativo)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            search_url = f"https://www.musixmatch.com/search/{artist_name} {song_title}"
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Aquí necesitarías más lógica para extraer letras específicas
                # Musixmatch tiene protección anti-scraping, así que este es básico
                
            return None
            
        except Exception as e:
            return None
    
    def translate_text(self, text, max_retries=3):
        """Traduce texto usando Google Translator"""
        try:
            # Dividir texto en chunks si es muy largo
            if len(text) > 4000:
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                translated_chunks = []
                for chunk in chunks:
                    translated = self.translator.translate(chunk)
                    translated_chunks.append(translated)
                    time.sleep(1)  # Rate limiting
                return ' '.join(translated_chunks)
            else:
                return self.translator.translate(text)
                
        except Exception as e:
            print(f"Error en traducción: {e}")
            if max_retries > 0:
                time.sleep(2)
                return self.translate_text(text, max_retries-1)
            return None
    
    def clean_lyrics(self, lyrics):
        """Limpia y formatea las letras"""
        if not lyrics:
            return None
            
        # Remover líneas vacías y espacios extras
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Filtrar líneas muy cortas (probablemente no sean letras)
        lines = [line for line in lines if len(line.split()) > 2]
        
        # Unir en párrafos de 4-8 líneas
        cleaned_lyrics = []
        current_paragraph = []
        
        for line in lines:
            current_paragraph.append(line)
            if len(current_paragraph) >= random.randint(3, 6):
                paragraph = ' '.join(current_paragraph)
                if len(paragraph) < 500:  # No párrafos muy largos
                    cleaned_lyrics.append(paragraph)
                current_paragraph = []
        
        # Añadir último párrafo si existe
        if current_paragraph:
            paragraph = ' '.join(current_paragraph)
            if len(paragraph) < 500:
                cleaned_lyrics.append(paragraph)
        
        return cleaned_lyrics
    
    def process_song(self, song_title, artist_name):
        """Procesa una canción completa"""
        try:
            # Obtener letras
            lyrics = self.get_genius_lyrics(song_title, artist_name)
            
            if not lyrics:
                lyrics = self.get_musixmatch_lyrics(song_title, artist_name)
            
            if not lyrics:
                return False
            
            # Limpiar y dividir letras
            cleaned_paragraphs = self.clean_lyrics(lyrics)
            
            if not cleaned_paragraphs:
                return False
            
            # Procesar cada párrafo
            for english_text in cleaned_paragraphs:
                if len(english_text) > 50:  # Mínimo de caracteres
                    # Traducir
                    spanish_text = self.translate_text(english_text)
                    
                    if spanish_text and len(spanish_text) > 30:
                        self.dataset.append({
                            'english': english_text,
                            'spanish': spanish_text,
                            'song': song_title,
                            'artist': artist_name
                        })
                        
                        print(f"✅ Añadido: {song_title} - {artist_name}")
                        print(f"EN: {english_text[:100]}...")
                        print(f"ES: {spanish_text[:100]}...")
                        print("-" * 50)
            
            return True
            
        except Exception as e:
            print(f"Error procesando canción {song_title}: {e}")
            return False
    
    def load_popular_songs(self):
        """Carga lista de canciones populares para scraping"""
        # Lista de artistas populares
        artists = [
            'Taylor Swift', 'Ed Sheeran', 'Adele', 'Beyonce', 'Drake',
            'The Beatles', 'Coldplay', 'Maroon 5', 'Rihanna', 'Justin Bieber',
            'Bruno Mars', 'Lady Gaga', 'Katy Perry', 'Eminem', 'Shakira',
            'Queen', 'Michael Jackson', 'Elton John', 'Madonna', 'Whitney Houston',
            'John Lennon', 'Bob Dylan', 'David Bowie', 'Prince', 'Stevie Wonder'
        ]
        
        # Canciones populares por artista (ejemplo simplificado)
        songs_by_artist = {
            'Taylor Swift': ['Love Story', 'Shake It Off', 'Blank Space', 'Bad Blood', 'You Belong With Me'],
            'Ed Sheeran': ['Shape of You', 'Perfect', 'Thinking Out Loud', 'Photograph', 'Castle on the Hill'],
            # Agrega más artistas y canciones...
        }
        
        songs_list = []
        for artist in artists:
            if artist in songs_by_artist:
                for song in songs_by_artist[artist]:
                    songs_list.append((song, artist))
            else:
                # Agregar algunas canciones genéricas
                songs_list.extend([
                    (f"Song {i}", artist) for i in range(1, 6)
                ])
        
        return songs_list
    
    def create_dataset(self, target_size=10000):
        """Crea el dataset completo"""
        songs_list = self.load_popular_songs()
        random.shuffle(songs_list)
        
        print(f"Iniciando creación de dataset con {len(songs_list)} canciones...")
        
        successful_songs = 0
        
        for song_title, artist_name in tqdm(songs_list):
            if len(self.dataset) >= target_size:
                break
            
            if self.process_song(song_title, artist_name):
                successful_songs += 1
            
            # Rate limiting para no saturar las APIs
            time.sleep(random.uniform(2, 5))
            
            # Guardar progreso cada 100 registros
            if len(self.dataset) % 100 == 0:
                self.save_dataset()
        
        self.save_dataset()
        print(f"Dataset completado! {len(self.dataset)} registros creados.")
    
    def save_dataset(self):
        """Guarda el dataset en archivos"""
        if self.dataset:
            df = pd.DataFrame(self.dataset)
            
            # Guardar en CSV
            df.to_csv('lyrics_dataset.csv', index=False, encoding='utf-8')
            
            # Guardar en JSON
            df.to_json('lyrics_dataset.json', orient='records', force_ascii=False)
            
            print(f"Dataset guardado: {len(self.dataset)} registros")
    
    def load_existing_dataset(self):
        """Carga dataset existente si existe"""
        try:
            if os.path.exists('lyrics_dataset.csv'):
                df = pd.read_csv('lyrics_dataset.csv')
                self.dataset = df.to_dict('records')
                print(f"Dataset cargado: {len(self.dataset)} registros")
                return True
        except:
            pass
        return False

# Ejecución principal
if __name__ == "__main__":
    creator = LyricsDatasetCreator()
    
    # Cargar dataset existente o crear nuevo
    if not creator.load_existing_dataset():
        print("Creando nuevo dataset...")
        creator.create_dataset(target_size=10000)
    else:
        print("Continuando con dataset existente...")
        # Continuar agregando más datos
        creator.create_dataset(target_size=10000)