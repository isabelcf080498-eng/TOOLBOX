## PRESENTACION_TOOLBOX


PRIMERA PARTE:

Speech:
Mostrar repo y abrir toolbox_ML.py.

- En este proyecto hicimos una toolbox en Python para acelerar un flujo muy típico de Machine Learning: entender el dataset y seleccionar variables cuando el objetivo
es una variable numérica, o sea, un caso de regresión.
La idea es que, en vez de repetir siempre el mismo código, tengamos funciones reutilizables. La toolbox está en el archivo toolbox_ML.py y tiene dos bloques:
funciones para describir y tipificar el dataset, y
funciones para encontrar features numéricas y categóricas relevantes respecto a un target, con métricas y tests estadísticos, y además visualizarlas.

- Primero está describe_df(df): devuelve un resumen por columna con tipo de dato, nulos, porcentaje de nulos y número de valores únicos. Lo útil es que en 10 segundos
detectamos columnas problemáticas, como cuando el porcentaje de nulos es muy alto.
Luego tipifica_variables(df, umbral_categoria, umbral_continua) sugiere si una variable es binaria, categórica o numérica. Esto es clave porque no analizas igual una
numérica que una categórica: para numéricas usaremos correlación, y para categóricas usaremos ANOVA.

- Después vienen las de regresión:
*get_features_num_regression:* recorre columnas numéricas y calcula correlación de Pearson con el target, quedándose con las que superan un umbral.
*plot_features_num_regression:* crea pairplots con Seaborn para ver esas relaciones de forma visual.
*get_features_cat_regression:* para categóricas aplica ANOVA, comparando medias del target entre grupos; si el p-value es menor al umbral, la categórica se considera
relevante.
y por ultimo *plot_features_cat_regression:* dibuja histogramas del target separados por categorías, para ver si realmente cambian las distribuciones.

Ahora lo vamos a demostrar con un dataset real, ejecutando los notebooks.”

__________________________________________________________________________________________________________________________________________________________________________

SEGUNDA PARTE
(Mini demo con prueba_toolbox (1).ipynb + lectura rápida del Titanic)

En pantalla (acciones):
Abrir prueba_toolbox (1).ipynb
Ejecutar celdas (Run All o de arriba abajo).
Enseñar df.head() y df.info() y el output de describe_df.

Speech:
“Para arrancar, hacemos una prueba rápida con el notebook prueba_toolbox. Importamos librerías y la toolbox, y cargamos el dataset Titanic desde una URL con pandas.read_csv.
Lo primero que enseñamos es df.head(), porque confirma qué columnas tenemos. En Titanic aparecen variables como Survived, Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, 
Cabin, Embarked… y ya podemos hacernos una idea de qué es numérico y qué es categórico.

- Luego corremos df.info() para ver: número de filas, tipos de datos y sobre todo valores no nulos. Aquí suele saltar lo típico del Titanic: Age tiene nulos, Cabin tiene
bastantes nulos y Embarked puede tener algunos. Esta parte es importante porque cualquier análisis serio empieza por saber qué tan incompletos están los datos.

- Después usamos describe_df(df). Y aquí lo que destacamos del output es:
el conteo de nulos por columna, el porcentaje de nulos, que te permite decidir si una columna se elimina o se imputa y los valores únicos, que sirve para detectar si una
 variable puede ser categórica o tiene demasiada cardinalidad.

- Con esto cerramos la prueba corta: ya confirmamos que la toolbox funciona, que carga datos reales y que nos da un diagnóstico inicial rápido.
Ahora pasamos al notebook largo donde está el desarrollo completo del ejemplo.”

___________________________________________________________________________________________________________________________________________________________________________

TERCERA PARTE
Desarrollo completo: tipificación + features numéricas con target Fare

Abrir Aplicacion_funciones.ipynb
Ejecutar celdas 0–10 (hasta visualización numérica).
Mostrar: describe_df, tipifica_variables, lista features_num, y 1 pairplot.

Speech:
“Ahora sí: el notebook Aplicacion_funciones es el desarrollo completo del ejemplo.
Repetimos imports y carga del Titanic. Primero ejecutamos df.info() y describe_df(df) para volver a ver lo mismo pero ya como parte del flujo: detectamos nulos en columnas
relevantes y confirmamos tipos.
Luego viene una parte clave: *tipifica_variables(df, umbral_categoria=10, umbral_continua=0.95).* Aquí el objetivo no es ‘adivinar perfecto’, sino tener una guía automática 
para decidir por dónde comenzar. Por ejemplo, columnas como *Sex* o *Embarked* salen como categóricas, y otras como *Fare* o *Age* como numéricas.

- A partir de aquí elegimos un target numérico: *Fare,* que es el precio del billete. Y buscamos qué variables numéricas podrían explicar su variación.

Ejecutamos *get_features_num_regression(df, target_col='Fare', umbral_corr=0.3)*. La función calcula correlación de Pearson entre cada columna numérica y *Fare.*
Luego imprime una lista de variables que cumplen el umbral y cuántas son. Esto no es causalidad, pero nos da un ranking rápido de variables con relación lineal con el target.

Después pasamos a lo visual con *plot_features_num_regression.* Lo importante es explicar qué se ve: los pairplots muestran dispersión y tendencia. Si ves una nube sin forma,
probablemente la relación es débil; si ves tendencia clara, hay relación. Además, la función divide el plotting en bloques para que no salga un gráfico inadecuadamente gigante.

Aquí mostramos uno de los plots y cerramos: ya tenemos un conjunto de features numéricas candidatas para modelar *Fare.*

_____________________________________________________________________________________________________________________________________________________________________________

CUARTA PARTE
Features categóricas con ANOVA + visualización + segundo target Age (flexibilidad)

En pantalla (acciones):
Ejecutar celdas 12–16.
Mostrar lista features_cat, 1 histplot, y luego repetir con Age.

Speech:
- Hasta ahora seleccionamos numéricas. Pero en Titanic hay variables categóricas muy relevantes para explicar un target numérico, por ejemplo *Pclass* o *Sex.*
Para eso usamos estadística: *ANOVA.*

- Ejecutamos get_features_cat_regression(df, target_col='Fare', p_value=0.05). ¿Qué hace?
Agrupa el target *Fare* por cada categoría, por ejemplo compara la distribución de *Fare* entre clases 1, 2 y 3; o por puertos de embarque. ANOVA nos dice si hay diferencias 
significativas entre medias. Si el p-value es menor que 0.05, asumimos que la variable categórica sí aporta información sobre el target.

- Luego ejecutamos plot_features_cat_regression. En pantalla aparecen histogramas del target con colores por categoría. Aquí es donde ‘se entiende’ el test: si las curvas
 están separadas o cambian mucho, se ve claramente la influencia de la categoría.

- Y para cerrar, demostramos que el flujo sirve para otro target: usamos Age. Repetimos lo mismo: primero features numéricas con un umbral más permisivo, por ejemplo 0.2, y
luego categóricas con *ANOVA.* Con el mismo toolbox y cambiando solo el target, podemos analizar diferentes problemas sin reescribir todo.

- Conclusión final: el ejemplo muestra un pipeline completo y reutilizable: diagnóstico → tipificación → selección numérica por correlación → selección categórica por ANOVA →
 visualización. En mejoras futuras, podríamos añadir imputación de nulos, encoding automático y un pipeline que entrene un modelo al final.”

____________________________________________________________________________________________________________________________________________________________________________
RECOMENDACIONES PARA LA PRESENTACION EN VIDEO:

En los gráficos (pairplot/histplot): enseña 1–2 máximo y comenta lo que significa; el resto dejamos que salga pero sin explicarlo uno por uno.

Mientras corre un plot (a veces tarda): aprovechamos para decir interpretación.

Si Cabin aparece con demasiados nulos, mencionarlo como ejemplo de columna que probablemente no usaríais sin tratamiento.

