#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================
ETAPA 3: PREPROCESAMIENTO DE DATOS
===============================================================
Incluye limpieza, transformación, codificación y 
generación de variables derivadas (Feature Engineering).
===============================================================
"""

import pandas as pd
import numpy as np

def preprocesar_datos(df):
    """
    Preprocesa los datos para el modelado.
    
    Args:
        df: DataFrame con los datos
    
    Returns:
        pd.DataFrame: Dataset preprocesado
    """
    print("=" * 50)
    print("ETAPA 3: PREPROCESAMIENTO")
    print("=" * 50)
    
    df_procesado = df.copy()
    
    # 3.1 Limpieza de datos
    print("\n3.1 LIMPIEZA DE DATOS")
    print("-" * 40)
    
    # Eliminar duplicados
    n_inicial = len(df_procesado)
    df_procesado = df_procesado.drop_duplicates()
    n_duplicados = n_inicial - len(df_procesado)
    print(f"✓ Duplicados eliminados: {n_duplicados}")
    
    # Manejo de valores nulos (si existieran)
    if df_procesado.isnull().sum().sum() > 0:
        for col in df_procesado.select_dtypes(include=[np.number]).columns:
            df_procesado[col].fillna(df_procesado[col].mean(), inplace=True)
        for col in df_procesado.select_dtypes(include=['object']).columns:
            df_procesado[col].fillna(df_procesado[col].mode()[0], inplace=True)
        print("✓ Valores nulos tratados")
    else:
        print("✓ No se requieren tratamientos adicionales")
    
    # 3.2 Transformaciones
    print("\n3.2 TRANSFORMACIONES")
    print("-" * 40)
    
    # Codificación de variables categóricas
    print("Codificación de variables categóricas:")
    
    # Tipo de tarjeta
    tipo_tarjeta_map = {'Credito': 0, 'Debito': 1}
    df_procesado['tipo_tarjeta_encoded'] = df_procesado['tipo_tarjeta'].map(tipo_tarjeta_map)
    print(f"  • tipo_tarjeta: {tipo_tarjeta_map}")
    
    # País
    pais_map = {'Local': 0, 'Internacional': 1}
    df_procesado['pais_encoded'] = df_procesado['pais'].map(pais_map)
    print(f"  • pais: {pais_map}")
    
    # Categoría
    categoria_map = {cat: i for i, cat in enumerate(df['categoria'].unique())}
    df_procesado['categoria_encoded'] = df_procesado['categoria'].map(categoria_map)
    print(f"  • categoria: {categoria_map}")
    
    # Normalización de variables numéricas (manual)
    print("\nNormalización de variables numéricas:")
    numeric_features = ['monto', 'numero_intentos', 'distancia_promedio', 'transacciones_ultima_hora']
    
    for col in numeric_features:
        mean = df_procesado[col].mean()
        std = df_procesado[col].std()
        df_procesado[col + '_normalized'] = (df_procesado[col] - mean) / std
        print(f"  • {col}: z = (x - {mean:.2f}) / {std:.2f}")
    
    # 3.3 Feature Engineering
    print("\n3.3 FEATURE ENGINEERING")
    print("-" * 40)
    
    # Variable 1: Índice de riesgo horario
    df_procesado['riesgo_horario'] = ((df_procesado['hora'] >= 22) | 
                                        (df_procesado['hora'] <= 5)).astype(int)
    print("✓ Variable derivada 1: 'riesgo_horario'")
    print("  Descripción: Identifica transacciones en horario nocturno (22:00-05:00)")
    print(f"  Transacciones en horario de riesgo: {df_procesado['riesgo_horario'].sum()} ({df_procesado['riesgo_horario'].mean()*100:.1f}%)")
    
    # Variable 2: Intensidad transaccional
    df_procesado['intensidad_transaccional'] = (
        df_procesado['transacciones_ultima_hora'] / 
        (df_procesado['numero_intentos'] + 1)
    )
    print("✓ Variable derivada 2: 'intensidad_transaccional'")
    print("  Fórmula: transacciones_ultima_hora / (numero_intentos + 1)")
    print(f"  Media: {df_procesado['intensidad_transaccional'].mean():.2f}")
    
    # Variable 3: Es fin de semana
    df_procesado['es_fin_semana'] = (df_procesado['dia_semana'] >= 5).astype(int)
    print("✓ Variable derivada 3: 'es_fin_semana'")
    print(f"  Transacciones en fin de semana: {df_procesado['es_fin_semana'].sum()} ({df_procesado['es_fin_semana'].mean()*100:.1f}%)")
    
    # Variable 4: Monto elevado
    monto_median = df_procesado['monto'].median()
    df_procesado['monto_elevado'] = (df_procesado['monto'] > monto_median).astype(int)
    print("✓ Variable derivada 4: 'monto_elevado'")
    print(f"  Umbral: > ${monto_median:.2f} (mediana)")
    print(f"  Transacciones con monto elevado: {df_procesado['monto_elevado'].sum()} ({df_procesado['monto_elevado'].mean()*100:.1f}%)")
    
    # Variable 5: Ratio de riesgo internacional
    df_procesado['riesgo_internacional'] = (df_procesado['pais'] == 'Internacional').astype(int)
    print("✓ Variable derivada 5: 'riesgo_internacional'")
    
    # 3.4 Resumen
    print("\n3.4 RESUMEN DEL PREPROCESAMIENTO")
    print("-" * 40)
    print(f"Registros finales: {len(df_procesado)}")
    print(f"Variables totales: {df_procesado.shape[1]}")
    print(f"Variables originales: {df.shape[1]}")
    print(f"Variables derivadas: {df_procesado.shape[1] - df.shape[1]}")
    
    # Guardar dataset preprocesado
    df_procesado.to_csv('datos_preprocesados.csv', index=False)
    print("\n✓ Dataset preprocesado guardado en 'datos_preprocesados.csv'")
    
    print("\n✓ Preprocesamiento completado")
    return df_procesado

if __name__ == "__main__":
    from download_dataset import descargar_dataset
    from eda import analisis_exploratorio
    df = descargar_dataset()
    df_analizado = analisis_exploratorio(df)
    preprocesar_datos(df_analizado)
