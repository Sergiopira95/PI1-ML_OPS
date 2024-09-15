# Importación de librerías 
from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# Inicializamos la aplicación FastAPI
app = FastAPI(title="& opsciones de consulta al alcance de un click", description="El mejor sistema de recomendación de pelis...",
               docs_url="/docs")
#Cargue de datos
df = pd.read_parquet('API_data.parquet')
model5 = pd.read_parquet('movies_model5.parquet')

#Página bienvenida
@app.get("/")
async def index():
    return "Bienvenid@s! Es hora de consultar cinefilos"

# Convertimos la columna de fechas a formato datetime para facilitar el manejo de fechas
df['release_date'] = pd.to_datetime(df['release_date'])

# 1. Cantidad de filmaciones por mes
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    # Diccionario para mapear los nombres de los meses en español a su número
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }

    # Convertimos el mes a minúsculas para asegurar coincidencia
    mes = mes.lower()

    # Verificamos si el mes ingresado es válido
    if mes not in meses:
        raise HTTPException(status_code=400, detail="Mes inválido. Use un mes en español, e.g., 'enero', 'febrero', etc.")

    # Filtramos el DataFrame por el mes solicitado
    mes_num = meses[mes]
    cantidad = df[df['release_date'].dt.month == mes_num].shape[0]

    return {f"La cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}": cantidad}

# 2. Cantidad de filmaciones por día
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    # Diccionario para mapear los días en español a su número (lunes es 0, domingo es 6)
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
    }

    # Convertimos el día a minúsculas para asegurar coincidencia
    dia = dia.lower()

    # Verificamos si el día ingresado es válido
    if dia not in dias:
        raise HTTPException(status_code=400, detail="Día inválido. Use un día en español, e.g., 'lunes', 'martes', etc.")

    # Filtramos el DataFrame por el día solicitado
    dia_num = dias[dia]
    cantidad = df[df['release_date'].dt.dayofweek == dia_num].shape[0]

    return {f"La cantidad de películas fueron estrenadas en los días {dia.capitalize()}": cantidad}

# 3. Score por título
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    # Buscamos la película por su título
    film = df[df['title'].str.lower() == titulo.lower()]

    if film.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    # Obtenemos el título, año de estreno y score
    titulo = film.iloc[0]['title']
    anio = film.iloc[0]['release_year']
    score = film.iloc[0]['vote_average']

    return {f"La película {titulo} fue estrenada en el año {anio} con un score de {score}"}

# 4. Votos por título
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    # Buscamos la película por su título
    film = df[df['title'].str.lower() == titulo.lower()]

    if film.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    # Obtenemos el número de votos y el promedio de los votos
    votos = film.iloc[0]['vote_count']
    promedio_votos = film.iloc[0]['vote_average']

    # Verificamos si la película tiene al menos 2000 valoraciones
    if votos < 2000:
        return {"mensaje": f"La película {titulo} no tiene suficientes valoraciones. Requiere al menos 2000."}

    return {f"La película {titulo} tiene {votos} votos y un promedio de {promedio_votos}"}

# 5. Información del éxito de un actor
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    # Filtramos el dataset para obtener las películas en las que ha participado el actor
    films_with_actor = df[df['actors'].str.contains(nombre_actor, case=False, na=False)]

    if films_with_actor.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado.")

    # Calculamos el total y el promedio de retorno para las películas en las que participó
    total_return = films_with_actor['return'].sum()
    promedio_return = films_with_actor['return'].mean()
    cantidad_peliculas = films_with_actor.shape[0]

    return {
        "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} filmaciones.",
        "total_retorno": total_return,
        "promedio_retorno": promedio_return
    }

# 6. Información del éxito de un director
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    # Filtramos el dataset para obtener las películas dirigidas por el director
    films_by_director = df[df['directors'].str.contains(nombre_director, case=False, na=False)]

    if films_by_director.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado.")

    # Para cada película, obtenemos el título, fecha de lanzamiento, retorno, costo y ganancia
    film_details = films_by_director[['title', 'release_date', 'return', 'budget', 'revenue']]

    # Convertimos el DataFrame a una lista de diccionarios para poder devolverla como JSON
    film_list = film_details.to_dict(orient='records')

    total_return = films_by_director['return'].sum()

    return {
        "mensaje": f"El director {nombre_director} ha dirigido {films_by_director.shape[0]} filmaciones.",
        "total_retorno": total_return,
        "peliculas": film_list
    }


#Machine Learning
model5['title'] = model5['title'].str.lower()
#Se separan los géneros y se convierten en palabras individuales
model5['genres'] = model5['genres'].fillna('').apply(lambda x: ' '.join(x.replace(',', ' ').replace('-', '').lower().split()))
#Se separan los slogans y se convierten en palabras individuales
model5['tagline'] = model5['tagline'].fillna('').apply(lambda x: ' '.join(x.replace(',', ' ').replace('-', '').lower().split()))
#Se crea una instancia de la clase TfidfVectorizer 
tfidf_5 = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
#Aplicar la transformación TF-IDF y obtener matriz numérica
tfidf_matriz_5 = tfidf_5.fit_transform(model5['genres'] + ' ' + model5['tagline'] + ' ' + model5['first_actor']+ ' ' + model5['first_director'])
#Función para obtener recomendaciones
@app.get('/recomendacion/{titulo}', name = "Sistema de recomendación")
async def recomendacion(titulo: str):
    '''Se ingresa el título de una película, por ejemplo "Avatar", y devuelve 5 recomendaciones.'''
    
    #Crear una serie que asigna un índice a cada título de las películas
    movies = pd.Series(model5.index, index=model5['title']).drop_duplicates()
    if titulo not in movies:
        return 'La película ingresada no se encuentra en la base de datos'
    else:
        #Obtener el índice de la película que coincide con el título
        ind = pd.Series(movies[titulo]) if titulo in movies else None
        #Si el título de la película está duplicado, devolver el índice de la primera aparición del título en el DataFrame
        if model5.duplicated(['title']).any():
            primer_ind = model5[model5['title'] == titulo].index[0]
            if not ind.equals(pd.Series(primer_ind)):
                ind = pd.Series(primer_ind)
        #Calcular la similitud coseno entre la película de entrada y todas las demás películas en la matriz de características
        cosine_sim = cosine_similarity(tfidf_matriz_5[ind], tfidf_matriz_5).flatten()
        simil = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)[1:6]
        #Verificar que los índices obtenidos son válidos
        valid_ind = [i[0] for i in simil if i[0] < len(model5)]
        #Obtener los títulos de las películas más similares utilizando el índice de cada película
        recomendaciones = model5.iloc[valid_ind]['title'].tolist()
        #Devolver la lista de títulos de las películas recomendadas
        return recomendaciones