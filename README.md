#<h1 align=center> 🚀**ETAPA DE LABS**🚀</h1>
# <h1 align=center> **PROYECTO INDIVIDUAL 1** </h1>
# <h2 align=center> **Sergio Andrés Piratoba Forero** </h2>

# <h1 align=center>**`Sistema de Recomendación para Plataforma de Streaming`**</h1>

![MLops](img/ML.webp)

## ```Introducción```
¡Bienvenido a este emocionante proyecto de práctica para el bootcamp de Data Science! 😎🚀

Contexto: he desarrollado un modelo de recomendación que ofrece métricas sobresalientes en tu entorno de pruebas. ¡Genial! Pero, ¿cómo llevar ese modelo a la vida real? 🌍👀 En el ciclo de vida de un proyecto de Machine Learning, no solo es crucial construir un modelo eficaz, sino también implementar un sistema que pueda integrar datos del mundo real, entrenar y mantener el modelo a medida que llegan nuevos datos.

En este proyecto, asumiré el rol de Data Scientist en una start-up innovadora que se especializa en la agregación de plataformas de streaming. Me enfrentare al reto de llevar a cabo mi primer modelo de ML para solucionar un problema de negocio real: un sistema de recomendación que aún no ha sido implementado.

Al sumergirme en los datos de la start-up, descubro que la madurez de los datos es, lamentablemente, nula 😭. Gran reto trabajarcon datos anidados y desordenados, sin procesos automatizados para la actualización de nuevas películas o series. En resumen, ¡tendré que hacer malabares entre las tareas de Data Engineering y Machine Learning! 😩🤯

Mi objetivo es empezar desde cero y construir un MVP (Minimum Viable Product) que sea funcional en las próximas semanas. Sí, el camino será desafiante y exigente, pero se qué hacer Con un enfoque claro y una estrategia bien definida, me enfrentaré a este reto con determinación 💪 y lograré convertir el modelo teórico en una herramienta valiosa para la start-up.

¡Estoy listo para trabajar duro y transformar datos crudos en un sistema de recomendación listo para el mundo real! 🌟📈
## :white_check_mark: ```Objetivo General```

- :pushpin: Realizar el ciclo completo de MLOps para obtener un API funcional que permita ejecutar funciones especificas y recomiede peliculas.

## :white_check_mark: ```Objetivos Específicos ```

- :pushpin: Llevar acabo el procesamiento de los datos para prepararlos y limpiarlos para la ingesta al modelo (ETL).
- :pushpin: Realizar el análisis exploratoorio de los datos para reconocer patrones que ayuden a tomar decisiones. (EDA).
- :pushpin: Crear endpoints de API que permitan consultas específicas y recomendacion de películas basada en el contenido.

## :white_check_mark: ```Metodología de trabajo```

Para llevar a cabo los objetivos, se ejecutaron los siguientes procedimientos:


- :one: ${\color{red} \textbf{Visual Studio Code}}$: se utilizó este editor de código para crear un directorio con el nombre del proyecto y se implementó un entorno virtual de forma local. En este entorno se procedió a crear la API junto a los endpoints. Para la construcción de la Api se utilizó el framework de Python FastAPI.

Los endpoints desarrollados fueron: 

- ```def cantidad_peliculas_mes(mes)```: Se ingresa el mes en minúscula, por ejemplo junio, y la función retorna la cantidad de películas que se estrenaron en ese mes
    
        Formato de salida: En el mes de {mes} se estrenaron {cantidad} películas

- ```def cantidad_peliculas_dia(dia)```: Se ingresa el día en minúscula, por ejemplo sábado, y la función retorna la cantidad de películas que se estrenaron ese día
    

        Formato de salida: En el día {dia} se estrenaron {cantidad} películas

- ```def score_titulo(titulo)```: Se ingresa el título de una película, por ejemplo "Avatar", y se retorna el título, el año de estreno y el score.
    

        Formato de salida: "Título de la película": resultado['title'], "Año": resultado['release_year'], "Puntaje": resultado['vote_average']

- ```def votos_titulo(titulo)```: Se ingresa el título de una película, por ejemplo "Fast and Furious", y se retorna el título, el año de estreno y el score.

        Formato de salida: {
                'Título de la película': titulo, 
                 'Año': year_es, 
                 'Voto total': voto_tot, 
                 'Voto promedio': voto_prom
                                  }

- ```def get_actor(nombre_actor)```: Se ingresa el nombre de un actor, por ejemplo "Jaime Foxx" y se retorna su éxito medido a través del retorno, cantidad de películas y promedio de retorno.
    

        Formato de salida: {
        "Actor/Actriz": nombre_actor,
        "Cantidad de películas": cantidad_peliculas,
        "Retorno Total": total_retorno,
        "Retorno Promedio": promedio_retorno}

- ```def get_director(nombre_director)```: Se ingresa el nombre de un director por ejemplo "Cristopher Nolan" y se retorna su éxito medido a través del retorno, nombre de cada película, fecha de lanzamiento, retorno individual, costo y ganancia.
    
        Formato de salida: {
        "Director": nombre_director,
        "Retorno Total": total_retorno,
        "Películas": resultado}
        {
            "Título de la película":   ,
            "Fecha de lanzamiento":    ,
            "Retorno":      ,
            "Presupuesto":   ,
            "Ganancia":    }


- :two: ${\color{red} \textbf{Visual Studio Code}}$: Se utilizó esta plataforma para el desarrollo de los procesos ETL, EDA y Modelo de Machine Learning. 
    - **ETL:** se realizó limpieza y transformación de los datos para garantizar la calidad y consistencia de la información utilizada en el sistema de recomendación. El resultado losjupyter notebooks (movies_datasetETL.ipynb y credits_dataset_ETL.ipynb) desarrollados para esta etapa corresponden al conjunto de datos que se utilizó para alimentar a la Api, se lo descargó en formato parquet con el nombre  API_data.parquet.
    - **EDA:** este análisis se realizó con la finalidad de identificar patrones, tendencias y relaciones en los datos, así como detectar posibles outliers y anomalías. Dicho análisis posibilitó decidir cuáles atributos eran los adecuados para aplicar el Modelo de Machine Learning. El resultado del jupyter notebook (movies_data_EDA.ipynb) desarrolado para esta etapa corresponde al conjunto de datos que se utilizó para aplicar el modelo seleccionado.
    - **Modelo de Machine Learning (ML):** para el modelado se seleccionaron TF-IDF (Term Frequency-Inverse Document Frequency) y la similitud del coseno. Estas son dos técnicas fundamentales que se utilizan en procesamiento de lenguaje natural (NLP) para medir la relevancia de términos en documentos y para calcular la similitud entre ellos. Los distintos modelos aplicados se encuentran en el notebook movies_ML.ipynb. Se eligió el modelo que responda al siguiente endpoint: 
       - def recomendacion(titulo)```: Se ingresa el título de una película, por ejemplo "Avatar", y devuelve 5 recomendaciones.
    

                Formato de salida: ['titulo_recomendado1', 'titulo_recomendado2', 'titulo_recomendado3', 'titulo_recomendado4', 'titulo_recomendado5']
    


- :three: ${\color{red} \textbf{Github}}$: se usó esta plataforma para almacenar el proyecto. Se creó un repositorió con el nombre **PI1-ML_OPS**. Este paso es imprescindible para deployar la Api en render, dado que se utiliza la dirección del repositorio para realizar el deploy. Cada cambio realizado a nivel local se iba actualizando en el repositorio

- :four: ${\color{red} \textbf{Render}}$: se utilizó este sitio para desplegar el proyecto, se creó una cuenta en el sitio y luego se conectó con el repositorio de Github donde se encuentra alojado el proyecto. Se tuvo que tener mucho cuidado al elegir el ML ya que render tiene un límite de memoria de 512 Mb.




## :white_check_mark: :sparkles: ```Deployment de la Api``` :sparkles:
 
 Para realizar consultas y recomendaciones de películas dirigirse a la siguiente dirección: [Movies Recomendation](https://pi1-ml-ops-deploy.onrender.com)




## :white_check_mark: ```Video```

El siguiente video muestra el funcionamiento de la Api, es útil como guía para que realices tus consultas: :clapper: [Video](https://drive.google.com/file/d/1pi_ZQif-E9qVCg0zF6DLeD449HOXsxz0/view?usp=sharing)

