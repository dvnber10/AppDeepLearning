import requests
def leer_txt(ruta_archivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
            return contenido
    except FileNotFoundError:
        print(f"⚠️ El archivo '{ruta_archivo}' no fue encontrado.")
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")

# Ejemplo de uso
ruta = 'output.txt'
texto = leer_txt(ruta)
if texto:
    print("Contenido del archivo:")
    print(texto)



url = "https://api.sunoapi.org/api/v1/generate"

payload = {
    "prompt": texto,
    "style": "Reggae, Upbeat Drums",
    "title": "cancion",
    "customMode": True,
    "instrumental": False,
    "model": "V3_5",
    "negativeTags": "",
    "vocalGender": "f",
    "styleWeight": 0.65,
    "weirdnessConstraint": 0.65,
    "audioWeight": 0.65,
    "callBackUrl": "playground"
}
headers = {
    "Authorization": "Bearer 7dc08faf09318c85f180244be1449f2d",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())