#!/usr/bin/env python3
"""
Script para inspeccionar la estructura del archivo de clustering
"""

import pickle
import numpy as np

def inspect_clustering_file():
    """Inspeccionar el archivo de clustering"""
    try:
        with open('clustering_results/clustering_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print("🔍 ESTRUCTURA DEL ARCHIVO DE CLUSTERING")
        print("="*50)
        print(f"Tipo: {type(results)}")
        
        if isinstance(results, dict):
            print(f"Claves disponibles: {list(results.keys())}")
            
            for key, value in results.items():
                print(f"\n📋 {key}:")
                print(f"   Tipo: {type(value)}")
                
                if isinstance(value, (list, tuple)):
                    print(f"   Longitud: {len(value)}")
                    if len(value) > 0:
                        print(f"   Primer elemento: {value[0]}")
                        print(f"   Tipo del primer elemento: {type(value[0])}")
                elif isinstance(value, np.ndarray):
                    print(f"   Forma: {value.shape}")
                    print(f"   Tipo de datos: {value.dtype}")
                elif isinstance(value, dict):
                    print(f"   Claves: {list(value.keys())}")
                else:
                    print(f"   Valor: {value}")
        else:
            print(f"Contenido: {results}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_clustering_file()
