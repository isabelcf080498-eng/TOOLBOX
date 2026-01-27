#Descargamos las herramientas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

def describe_df(df):
    """
    Da un vistazo rápido al dataframe: tipos, nulos, valores únicos y porcentaje de nulos.
    """
    # Creamos un resumen con el tipo de cada columna
    resumen = pd.DataFrame(df.dtypes, columns=['Tipo de Dato'])
    
    # Calculamos nulos y su porcentaje para saber qué tan "limpia" está la columna
    resumen['Nulos'] = df.isnull().sum()
    resumen['% Nulos'] = (df.isnull().sum() / len(df)) * 100
    resumen['Valores Únicos'] = df.nunique()
    
    return resumen.T

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de variable según su cardinalidad.
    """
    sugerencias = []
    for col in df.columns:
        cardinalidad = df[col].nunique()
        if cardinalidad == 2:
            sugerencias.append("Binaria")
        elif cardinalidad < umbral_categoria:
            sugerencias.append("Categórica")
        else:
            if cardinalidad >= umbral_continua:
                sugerencias.append("Numérica Continua")
            else:
                sugerencias.append("Numérica Discreta")
    return pd.DataFrame({'Variable': df.columns, 'Sugerencia': sugerencias})

def get_features_num_regression(df, target_col, umbral_corr, p_value=0.05):
    """
    Busca variables numéricas que tengan buena correlación con el target.
    """
    if target_col not in df.columns:
        return None
    
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    features_interesantes = []
    
    for col in cols_numericas:
        if col == target_col: continue
        correlacion = df[col].corr(df[target_col])
        if abs(correlacion) >= umbral_corr:
            features_interesantes.append(col)
            
    return features_interesantes

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, p_value=0.05):
    """
    Pinta los pairplots. Si no le das columnas, las busca él mismo.
    """
    # Si la lista está vacía, usamos nuestra función de filtrado
    if not columns:
        columns = get_features_num_regression(df, target_col, umbral_corr, p_value)
    
    if not columns:
        return []

    # Pintamos de 5 en 5 para que no se amontone todo
    for i in range(0, len(columns), 5):
        grupo = columns[i:i+5]
        sns.pairplot(df[grupo + [target_col]])
        plt.show()
        
    return columns

def get_features_cat_regression(df, target_col, p_value=0.05):
    """
    Usa ANOVA para ver qué variables categóricas influyen en el target.
    """
    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
    features_clave = []
    
    for col in cols_categoricas:
        # Quitamos nulos para que el test no de error
        temp_df = df[[col, target_col]].dropna()
        grupos = [group[target_col].values for name, group in temp_df.groupby(col)]
        
        if len(grupos) > 1:
            stat, p = f_oneway(*grupos)
            if p < p_value:
                features_clave.append(col)
                
    return features_clave

def plot_features_cat_regression(df, target_col="", columns=[], p_value=0.05):
    """
    Pinta histogramas de las categorías que afectan al target.
    """
    # Si no nos pasan columnas, las buscamos con el test estadístico
    if not columns:
        columns = get_features_cat_regression(df, target_col, p_value)
    
    if not columns:
        print("No se encontraron columnas categóricas significativas.")
        return []

    for col in columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=target_col, hue=col, kde=True, element="step")
        plt.title(f"Influencia de '{col}' en '{target_col}'")
        plt.show()
        
    return columns