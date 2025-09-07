"""
Módulo para análisis temporal y local vs global
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import chi2_contingency
import logging

logger = logging.getLogger(__name__)

def analyze_temporal_trends(metadata: pd.DataFrame,
                          cluster_labels: np.ndarray,
                          time_column: str = 'timestamp',
                          time_format: str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Analiza tendencias temporales de clusters
    
    Args:
        metadata: DataFrame con metadatos
        cluster_labels: Etiquetas de cluster
        time_column: Nombre de la columna de tiempo
        time_format: Formato de fecha
    
    Returns:
        DataFrame con tendencias temporales
    """
    try:
        # Crear DataFrame con datos temporales
        df = metadata.copy()
        df['cluster_id'] = cluster_labels
        
        # Convertir timestamp a datetime
        df[time_column] = pd.to_datetime(df[time_column], format=time_format, errors='coerce')
        
        # Filtrar datos válidos
        df = df.dropna(subset=[time_column])
        
        # Agrupar por mes
        df['month'] = df[time_column].dt.to_period('M')
        
        # Calcular proporciones por mes
        monthly_counts = df.groupby(['month', 'cluster_id']).size().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_proportions = monthly_counts.div(monthly_totals, axis=0)
        
        # Resetear índice para tener 'month' como columna
        monthly_proportions = monthly_proportions.reset_index()
        monthly_proportions['month'] = monthly_proportions['month'].astype(str)
        
        logger.info(f"Tendencias temporales calculadas para {len(monthly_proportions)} meses")
        return monthly_proportions
        
    except Exception as e:
        logger.error(f"Error analizando tendencias temporales: {e}")
        return pd.DataFrame()

def compare_local_global(metadata: pd.DataFrame,
                        cluster_labels: np.ndarray,
                        source_column: str = 'source') -> Dict[str, Any]:
    """
    Compara distribución de clusters entre fuentes local y global
    
    Args:
        metadata: DataFrame con metadatos
        cluster_labels: Etiquetas de cluster
        source_column: Nombre de la columna de fuente
    
    Returns:
        Diccionario con estadísticas de comparación
    """
    try:
        # Crear DataFrame con datos
        df = metadata.copy()
        df['cluster_id'] = cluster_labels
        
        # Filtrar datos válidos
        df = df.dropna(subset=[source_column])
        
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(df[source_column], df['cluster_id'])
        
        # Calcular estadísticas
        n_local = len(df[df[source_column] == 'local'])
        n_global = len(df[df[source_column] == 'global'])
        
        # Proporciones por fuente
        local_proportions = contingency_table.loc['local'] / n_local if 'local' in contingency_table.index else pd.Series()
        global_proportions = contingency_table.loc['global'] / n_global if 'global' in contingency_table.index else pd.Series()
        
        # Test de chi-cuadrado
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        except:
            chi2, p_value, dof, expected = 0, 1, 0, None
        
        # Diferencias significativas
        significant_differences = []
        for cluster_id in contingency_table.columns:
            if cluster_id in local_proportions.index and cluster_id in global_proportions.index:
                local_prop = local_proportions[cluster_id]
                global_prop = global_proportions[cluster_id]
                diff = abs(local_prop - global_prop)
                
                if diff > 0.1:  # Diferencia mayor al 10%
                    significant_differences.append({
                        'cluster_id': cluster_id,
                        'local_proportion': float(local_prop),
                        'global_proportion': float(global_prop),
                        'difference': float(diff)
                    })
        
        # Clusters más representativos por fuente
        local_dominant = local_proportions.idxmax() if not local_proportions.empty else None
        global_dominant = global_proportions.idxmax() if not global_proportions.empty else None
        
        results = {
            'contingency_table': contingency_table.to_dict(),
            'n_local': int(n_local),
            'n_global': int(n_global),
            'local_proportions': local_proportions.to_dict(),
            'global_proportions': global_proportions.to_dict(),
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant_differences': significant_differences,
            'local_dominant_cluster': int(local_dominant) if local_dominant is not None else None,
            'global_dominant_cluster': int(global_dominant) if global_dominant is not None else None,
            'is_significant': p_value < 0.05
        }
        
        logger.info(f"Comparación local vs global: chi2={chi2:.3f}, p={p_value:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Error comparando local vs global: {e}")
        return {}

def detect_trending_clusters(trends_data: pd.DataFrame,
                           threshold: float = 0.1,
                           min_months: int = 3) -> List[Dict[str, Any]]:
    """
    Detecta clusters con tendencias crecientes o decrecientes
    
    Args:
        trends_data: DataFrame con tendencias temporales
        threshold: Umbral de cambio para considerar tendencia
        min_months: Mínimo de meses para considerar tendencia
    
    Returns:
        Lista de clusters con tendencias
    """
    try:
        trending_clusters = []
        
        # Obtener columnas de clusters (excluyendo 'month')
        cluster_columns = [col for col in trends_data.columns if col != 'month']
        
        for cluster_col in cluster_columns:
            cluster_id = cluster_col
            proportions = trends_data[cluster_col].values
            
            # Calcular tendencia (pendiente de regresión lineal simple)
            if len(proportions) >= min_months:
                x = np.arange(len(proportions))
                slope = np.polyfit(x, proportions, 1)[0]
                
                # Determinar tipo de tendencia
                if slope > threshold:
                    trend_type = 'creciente'
                elif slope < -threshold:
                    trend_type = 'decreciente'
                else:
                    trend_type = 'estable'
                
                # Calcular métricas adicionales
                mean_proportion = np.mean(proportions)
                std_proportion = np.std(proportions)
                max_proportion = np.max(proportions)
                min_proportion = np.min(proportions)
                
                trending_clusters.append({
                    'cluster_id': cluster_id,
                    'trend_type': trend_type,
                    'slope': float(slope),
                    'mean_proportion': float(mean_proportion),
                    'std_proportion': float(std_proportion),
                    'max_proportion': float(max_proportion),
                    'min_proportion': float(min_proportion),
                    'volatility': float(std_proportion / mean_proportion) if mean_proportion > 0 else 0
                })
        
        # Ordenar por volatilidad descendente
        trending_clusters.sort(key=lambda x: x['volatility'], reverse=True)
        
        logger.info(f"Detectadas {len(trending_clusters)} tendencias de clusters")
        return trending_clusters
        
    except Exception as e:
        logger.error(f"Error detectando clusters tendenciosos: {e}")
        return []

def analyze_seasonal_patterns(trends_data: pd.DataFrame,
                            cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Analiza patrones estacionales en los clusters
    
    Args:
        trends_data: DataFrame con tendencias temporales
        cluster_labels: Etiquetas de cluster
    
    Returns:
        Diccionario con análisis estacional
    """
    try:
        # Convertir month a datetime para extraer estaciones
        trends_data['month_dt'] = pd.to_datetime(trends_data['month'])
        trends_data['season'] = trends_data['month_dt'].dt.month.map({
            12: 'invierno', 1: 'invierno', 2: 'invierno',
            3: 'primavera', 4: 'primavera', 5: 'primavera',
            6: 'verano', 7: 'verano', 8: 'verano',
            9: 'otoño', 10: 'otoño', 11: 'otoño'
        })
        
        # Calcular promedios por estación
        seasonal_analysis = {}
        cluster_columns = [col for col in trends_data.columns if col not in ['month', 'month_dt', 'season']]
        
        for cluster_col in cluster_columns:
            cluster_id = cluster_col
            seasonal_means = trends_data.groupby('season')[cluster_col].mean()
            
            seasonal_analysis[cluster_id] = {
                'invierno': float(seasonal_means.get('invierno', 0)),
                'primavera': float(seasonal_means.get('primavera', 0)),
                'verano': float(seasonal_means.get('verano', 0)),
                'otoño': float(seasonal_means.get('otoño', 0))
            }
        
        # Encontrar clusters más estacionales
        seasonal_clusters = []
        for cluster_id, seasons in seasonal_analysis.items():
            values = list(seasons.values())
            if values:
                seasonal_variance = np.var(values)
                dominant_season = max(seasons.keys(), key=lambda k: seasons[k])
                
                seasonal_clusters.append({
                    'cluster_id': cluster_id,
                    'variance': float(seasonal_variance),
                    'dominant_season': dominant_season,
                    'seasonal_values': seasons
                })
        
        # Ordenar por varianza estacional
        seasonal_clusters.sort(key=lambda x: x['variance'], reverse=True)
        
        results = {
            'seasonal_analysis': seasonal_analysis,
            'seasonal_clusters': seasonal_clusters,
            'most_seasonal': seasonal_clusters[0] if seasonal_clusters else None
        }
        
        logger.info(f"Análisis estacional completado para {len(seasonal_analysis)} clusters")
        return results
        
    except Exception as e:
        logger.error(f"Error analizando patrones estacionales: {e}")
        return {}

def create_temporal_summary(trends_data: pd.DataFrame,
                          local_global_stats: Dict[str, Any],
                          trending_clusters: List[Dict[str, Any]],
                          seasonal_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea resumen temporal completo
    
    Args:
        trends_data: DataFrame con tendencias temporales
        local_global_stats: Estadísticas local vs global
        trending_clusters: Clusters con tendencias
        seasonal_analysis: Análisis estacional
    
    Returns:
        Diccionario con resumen temporal
    """
    try:
        summary = {
            'temporal_analysis': {
                'total_months': len(trends_data),
                'date_range': {
                    'start': trends_data['month'].iloc[0] if not trends_data.empty else None,
                    'end': trends_data['month'].iloc[-1] if not trends_data.empty else None
                }
            },
            'local_global_comparison': {
                'is_significant': local_global_stats.get('is_significant', False),
                'p_value': local_global_stats.get('p_value', 1.0),
                'n_local': local_global_stats.get('n_local', 0),
                'n_global': local_global_stats.get('n_global', 0)
            },
            'trending_clusters': {
                'total': len(trending_clusters),
                'increasing': len([c for c in trending_clusters if c['trend_type'] == 'creciente']),
                'decreasing': len([c for c in trending_clusters if c['trend_type'] == 'decreciente']),
                'stable': len([c for c in trending_clusters if c['trend_type'] == 'estable'])
            },
            'seasonal_patterns': {
                'total_clusters': len(seasonal_analysis.get('seasonal_analysis', {})),
                'most_seasonal': seasonal_analysis.get('most_seasonal', {}).get('cluster_id'),
                'seasonal_variance': seasonal_analysis.get('most_seasonal', {}).get('variance', 0)
            }
        }
        
        logger.info("Resumen temporal creado")
        return summary
        
    except Exception as e:
        logger.error(f"Error creando resumen temporal: {e}")
        return {}


