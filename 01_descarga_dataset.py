#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================
ETAPA 1: DESCARGA Y CARGA DE DATOS
===============================================================
Dataset: Transacciones Bancarias con Indicadores de Fraude
===============================================================
"""

import pandas as pd
import numpy as np

def descargar_dataset():
    """
    Genera un dataset simulado de transacciones bancarias.
    En producción, esto reemplazaría con la carga de datos reales.
    
    Returns:
        pd.DataFrame: Dataset con transacciones bancarias
    """
    print("=" * 50)
    print("ETAPA 1: DESCARGA DE DATOS")
    print("=" * 50)
    
    np.random.seed(42)
    n_transacciones = 5000
    
    # Generación de dataset
    data = {
        'id_transaccion': range(1, n_transacciones + 1),
        'monto': np.random.exponential(scale=150, size=n_transacciones),
        'hora': np.random.randint(0, 24, n_transacciones),
        'dia_semana': np.random.randint(0, 7, n_transacciones),
        'tipo_tarjeta': np.random.choice(['Credito', 'Debito'], n_transacciones),
        'pais': np.random.choice(['Local', 'Internacional'], n_transacciones, p=[0.85, 0.15]),
        'categoria': np.random.choice(['Supermercado', 'Restaurante', 'Tienda', 'Online', 'ATM', 'Farmacia'], n_transacciones),
        'numero_intentos': np.random.poisson(lam=1, size=n_transacciones),
        'distancia_promedio': np.random.exponential(scale=50, size=n_transacciones),
        'transacciones_ultima_hora': np.random.poisson(lam=3, size=n_transacciones)
    }
    
    df = pd.DataFrame(data)
    
    # Crear variable objetivo (fraude) - correlacionada con características específicas
    prob_fraude = (
        (df['monto'] > 500) * 0.3 +
        (df['numero_intentos'] > 3) * 0.2 +
        (df['pais'] == 'Internacional') * 0.15 +
        (df['transacciones_ultima_hora'] > 5) * 0.2 +
        (df['hora'].isin([2, 3, 4, 5])) * 0.1
    )
    df['fraude'] = (np.random.random(n_transacciones) < np.clip(prob_fraude, 0, 0.9)).astype(int)
    
    # Asegurar proporción aproximada de fraudes (~2%)
    n_fraudes_deseado = int(n_transacciones * 0.02)
    if df['fraude'].sum() > n_fraudes_deseado:
        indices_fraude = df[df['fraude'] == 1].sample(df['fraude'].sum() - n_fraudes_deseado).index
        df.loc[indices_fraude, 'fraude'] = 0
    
    print(f"✓ Dataset generado: {df.shape[0]} registros, {df.shape[1]} variables")
    print(f"✓ Fraudes: {df['fraude'].sum()} ({df['fraude'].mean()*100:.2f}%)")
    print(f"✓ Transacciones legítimas: {(df['fraude']==0).sum()} ({(df['fraude']==0).mean()*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    df = descargar_dataset()
    print("\nPrimeras 5 filas:")
    print(df.head())
