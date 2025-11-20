# Sistema de Recomendación Básico con Álgebra Vectorial
# Integrantes: Gerald Delgado, Matias Lutz, Axel Ulate, Marco Alvarado

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class SistemaRecomendacion:
    def __init__(self):
        self.matriz_usuario_pelicula = None
        self.similitud_usuarios = None
        self.similitud_peliculas = None
        self.modelo_mf = None
        
    def cargar_datos(self, ruta_archivo):
        """
        Cargar y preparar datos del dataset MovieLens
        """
        try:
            # Cargar ratings
            ratings = pd.read_csv(ruta_archivo)
            print("Datos cargados exitosamente")
            print(f"Shape: {ratings.shape}")
            print(ratings.head())
            return ratings
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None
    
    def preprocesar_datos(self, ratings):
        """
        Preprocesamiento: limpieza, normalización y split
        """
        # Eliminar valores nulos
        ratings_clean = ratings.dropna()
        
        # Crear matriz usuario-película
        self.matriz_usuario_pelicula = ratings_clean.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        print(f"Matriz usuario-película creada: {self.matriz_usuario_pelicula.shape}")
        
        # Normalizar ratings (escala 0-1)
        ratings_normalized = (ratings_clean['rating'] - ratings_clean['rating'].min()) / \
                           (ratings_clean['rating'].max() - ratings_clean['rating'].min())
        
        # Split train-test
        train_data, test_data = train_test_split(ratings_clean, test_size=0.2, random_state=42)
        
        return train_data, test_data, self.matriz_usuario_pelicula
    
    def calcular_similitud_coseno(self, matriz):
        """
        Calcular similitud coseno entre vectores
        Fórmula: cos(θ) = (A·B) / (||A|| * ||B||)
        """
        similitud = cosine_similarity(matriz)
        return similitud
    
    def filtrado_colaborativo_usuario(self, usuario_id, k_vecinos=5, n_recomendaciones=10):
        """
        Filtrado colaborativo basado en usuario (UCF)
        """
        if self.similitud_usuarios is None:
            self.similitud_usuarios = self.calcular_similitud_coseno(self.matriz_usuario_pelicula)
        
        # Obtener usuarios más similares
        similitudes_usuario = self.similitud_usuarios[usuario_id]
        usuarios_similares = np.argsort(similitudes_usuario)[::-1][1:k_vecinos+1]
        
        # Calificar películas no vistas basado en vecinos
        calificaciones_usuario = self.matriz_usuario_pelicula.iloc[usuario_id]
        peliculas_no_vistas = calificaciones_usuario[calificaciones_usuario == 0].index
        
        predicciones = {}
        for pelicula in peliculas_no_vistas:
            # Promedio ponderado por similitud
            ratings_vecinos = []
            similitudes = []
            
            for vecino in usuarios_similares:
                rating_vecino = self.matriz_usuario_pelicula.iloc[vecino, pelicula]
                if rating_vecino > 0:
                    ratings_vecinos.append(rating_vecino)
                    similitudes.append(self.similitud_usuarios[usuario_id, vecino])
            
            if ratings_vecinos:
                predicciones[pelicula] = np.average(ratings_vecinos, weights=similitudes)
        
        # Top N recomendaciones
        recomendaciones = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:n_recomendaciones]
        return recomendaciones
    
    def filtrado_colaborativo_item(self, usuario_id, k_vecinos=5, n_recomendaciones=10):
        """
        Filtrado colaborativo basado en ítems (ICF)
        """
        if self.similitud_peliculas is None:
            matriz_transpuesta = self.matriz_usuario_pelicula.T
            self.similitud_peliculas = self.calcular_similitud_coseno(matriz_transpuesta)
        
        calificaciones_usuario = self.matriz_usuario_pelicula.iloc[usuario_id]
        peliculas_vistas = calificaciones_usuario[calificaciones_usuario > 0].index
        peliculas_no_vistas = calificaciones_usuario[calificaciones_usuario == 0].index
        
        predicciones = {}
        for pelicula in peliculas_no_vistas:
            # Encontrar películas similares que el usuario ha visto
            peliculas_similares = []
            similitudes = []
            
            for pelicula_vista in peliculas_vistas:
                similitud = self.similitud_peliculas[pelicula, pelicula_vista]
                if similitud > 0:
                    peliculas_similares.append(pelicula_vista)
                    similitudes.append(similitud)
            
            # Tomar las k películas más similares
            if len(peliculas_similares) > k_vecinos:
                indices_top = np.argsort(similitudes)[::-1][:k_vecinos]
                peliculas_similares = [peliculas_similares[i] for i in indices_top]
                similitudes = [similitudes[i] for i in indices_top]
            
            # Calcular predicción
            ratings_similares = [self.matriz_usuario_pelicula.iloc[usuario_id, p] for p in peliculas_similares]
            
            if ratings_similares:
                predicciones[pelicula] = np.average(ratings_similares, weights=similitudes)
        
        recomendaciones = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:n_recomendaciones]
        return recomendaciones
    
    def factorizacion_matrices(self, n_factores=10, metodo='svd'):
        """
        Factorización de Matrices usando SVD o NMF
        """
        matriz = self.matriz_usuario_pelicula.values
        
        if metodo == 'svd':
            self.modelo_mf = TruncatedSVD(n_components=n_factores, random_state=42)
        else:  # nmf
            self.modelo_mf = NMF(n_components=n_factores, random_state=42)
        
        matriz_latente = self.modelo_mf.fit_transform(matriz)
        return matriz_latente
    
    def predecir_mf(self, usuario_id, n_recomendaciones=10):
        """
        Predecir ratings usando factorización de matrices
        Fórmula: r_ui ≈ p_u · q_i
        """
        if self.modelo_mf is None:
            print("Primero debe entrenar el modelo de factorización")
            return None
        
        matriz_reconstruida = self.modelo_mf.inverse_transform(
            self.modelo_mf.transform(self.matriz_usuario_pelicula.values)
        )
        
        predicciones_usuario = matriz_reconstruida[usuario_id]
        calificaciones_reales = self.matriz_usuario_pelicula.iloc[usuario_id].values
        
        # Encontrar películas no vistas con alta predicción
        peliculas_no_vistas = np.where(calificaciones_reales == 0)[0]
        predicciones_no_vistas = [(i, predicciones_usuario[i]) for i in peliculas_no_vistas]
        
        recomendaciones = sorted(predicciones_no_vistas, key=lambda x: x[1], reverse=True)[:n_recomendaciones]
        return recomendaciones
    
    def evaluar_modelo(self, test_data, metodo='ucf', k_vecinos=5):
        """
        Evaluar modelo usando RMSE y MAE
        """
        predicciones = []
        reales = []
        
        for _, fila in test_data.iterrows():
            usuario_id = int(fila['userId']) - 1  # Ajustar índice
            pelicula_id = fila['movieId']
            rating_real = fila['rating']
            
            try:
                if metodo == 'ucf':
                    # Para UCF, obtener predicción del usuario para la película específica
                    rec = self.filtrado_colaborativo_usuario(usuario_id, k_vecinos, n_recomendaciones=100)
                    pred_dict = dict(rec)
                    rating_pred = pred_dict.get(pelicula_id, 0)
                elif metodo == 'icf':
                    rec = self.filtrado_colaborativo_item(usuario_id, k_vecinos, n_recomendaciones=100)
                    pred_dict = dict(rec)
                    rating_pred = pred_dict.get(pelicula_id, 0)
                else:  # mf
                    if self.modelo_mf is not None:
                        matriz_reconstruida = self.modelo_mf.inverse_transform(
                            self.modelo_mf.transform(self.matriz_usuario_pelicula.values)
                        )
                        rating_pred = matriz_reconstruida[usuario_id, pelicula_id]
                    else:
                        rating_pred = 0
                
                if rating_pred > 0:
                    predicciones.append(rating_pred)
                    reales.append(rating_real)
                    
            except Exception as e:
                continue
        
        if predicciones:
            rmse = np.sqrt(mean_squared_error(reales, predicciones))
            mae = mean_absolute_error(reales, predicciones)
            return rmse, mae
        else:
            return None, None
    
    def visualizar_resultados(self, resultados_rmse, resultados_mae):
        """
        Visualizar resultados de evaluación
        """
        metodos = list(resultados_rmse.keys())
        rmse_values = [resultados_rmse[m] for m in metodos]
        mae_values = [resultados_mae[m] for m in metodos]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico RMSE
        ax1.bar(metodos, rmse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Comparación de RMSE por Método')
        ax1.set_ylabel('RMSE')
        
        # Gráfico MAE
        ax2.bar(metodos, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Comparación de MAE por Método')
        ax2.set_ylabel('MAE')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso del sistema
def main():
    # Inicializar sistema
    sistema = SistemaRecomendacion()
    
    # Cargar datos (asumiendo que tenemos un archivo ratings.csv)
    # ratings = sistema.cargar_datos('ratings.csv')
    
    # Para propósitos de demostración, creamos datos de ejemplo
    np.random.seed(42)
    n_usuarios = 100
    n_peliculas = 50
    
    # Crear matriz usuario-película de ejemplo
    matriz_ejemplo = np.random.randint(0, 6, size=(n_usuarios, n_peliculas))
    usuarios = [f'Usuario_{i+1}' for i in range(n_usuarios)]
    peliculas = [f'Pelicula_{i+1}' for i in range(n_peliculas)]
    
    ratings = pd.DataFrame(matriz_ejemplo, index=usuarios, columns=peliculas)
    ratings_stacked = ratings.stack().reset_index()
    ratings_stacked.columns = ['userId', 'movieId', 'rating']
    ratings_stacked['userId'] = ratings_stacked['userId'].astype('category').cat.codes
    ratings_stacked['movieId'] = ratings_stacked['movieId'].astype('category').cat.codes
    
    # Preprocesar datos
    train_data, test_data, matriz = sistema.preprocesar_datos(ratings_stacked)
    sistema.matriz_usuario_pelicula = matriz
    
    print("=== SISTEMA DE RECOMENDACIÓN CON ÁLGEBRA VECTORIAL ===")
    
    # Probar diferentes métodos
    usuario_ejemplo = 0
    
    print(f"\n1. Filtrado Colaborativo Basado en Usuario (UCF):")
    rec_ucf = sistema.filtrado_colaborativo_usuario(usuario_ejemplo, k_vecinos=3)
    print(f"Recomendaciones para usuario {usuario_ejemplo}: {rec_ucf[:5]}")
    
    print(f"\n2. Filtrado Colaborativo Basado en Ítems (ICF):")
    rec_icf = sistema.filtrado_colaborativo_item(usuario_ejemplo, k_vecinos=3)
    print(f"Recomendaciones para usuario {usuario_ejemplo}: {rec_icf[:5]}")
    
    print(f"\n3. Factorización de Matrices (SVD):")
    sistema.factorizacion_matrices(n_factores=5, metodo='svd')
    rec_mf = sistema.predecir_mf(usuario_ejemplo)
    print(f"Recomendaciones para usuario {usuario_ejemplo}: {rec_mf[:5]}")
    
    # Evaluar modelos
    print(f"\n4. Evaluación de Modelos:")
    resultados_rmse = {}
    resultados_mae = {}
    
    for metodo in ['ucf', 'icf', 'mf']:
        rmse, mae = sistema.evaluar_modelo(test_data, metodo=metodo, k_vecinos=3)
        if rmse is not None:
            resultados_rmse[metodo.upper()] = rmse
            resultados_mae[metodo.upper()] = mae
            print(f"{metodo.upper()} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Visualizar resultados
    if resultados_rmse:
        sistema.visualizar_resultados(resultados_rmse, resultados_mae)
    
    print("\n=== IMPLEMENTACIÓN COMPLETADA ===")

if __name__ == "__main__":
    main()