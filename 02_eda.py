#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================
ETAPA 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
===============================================================
Incluye estadísticas descriptivas, valores nulos, duplicados
y detección de valores atípicos.
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def analisis_exploratorio(df):
    """
    Realiza el análisis exploratorio de datos.
    
    Args:
        df: DataFrame con los datos
    
    Returns:
        pd.DataFrame: Dataset analizado (sin cambios)
    """
    print("=" * 50)
    print("ETAPA 2: ANÁLISIS EXPLORATORIO")
    print("=" * 50)
    
    # Crear directorio para gráficos
    os.makedirs('graficos', exist_ok=True)
    
    # 2.1 Estadísticas descriptivas
    print("\n2.1 ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 40)
    print(df.describe())
    
    # 2.2 Reporte de valores nulos
    print("\n2.2 VALORES NULOS")
    print("-" * 40)
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("✓ No se encontraron valores nulos")
    else:
        print(nulos[nulos > 0])
    
    # 2.3 Registros duplicados
    print("\n2.3 REGISTROS DUPLICADOS")
    print("-" * 40)
    duplicados = df.duplicated().sum()
    print(f"Duplicados: {duplicados} ({(duplicados/len(df))*100:.2f}%)")
    
    # 2.4 Detección de valores atípicos (IQR)
    print("\n2.4 VALORES ATÍPICOS (IQR)")
    print("-" * 40)
    
    def detectar_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower) | (data[column] > upper)][column]
        return len(outliers), len(outliers)/len(data)*100
    
    for col in df.select_dtypes(include=[np.number]).columns:
        n_outliers, pct = detectar_outliers_iqr(df, col)
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} ({pct:.2f}%)")
    
    # 2.5 Distribución de la variable objetivo
    print("\n2.5 DISTRIBUCIÓN DE FRAUDE")
    print("-" * 40)
    print(df['fraude'].value_counts())
    print(f"\nTasa de fraude: {df['fraude'].mean()*100:.2f}%")
    
    # ============================================================
    # VISUALIZACIONES
    # ============================================================
    print("\n2.6 GENERANDO VISUALIZACIONES...")
    print("-" * 40)
    
    # Figura 1: Análisis Exploratorio
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 Histograma de montos
    axes[0, 0].hist(df['monto'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].set_title('Distribución de Montos', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Monto ($)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].axvline(df['monto'].mean(), color='red', linestyle='--', label=f'Media: ${df["monto"].mean():.2f}')
    axes[0, 0].legend()
    
    # 1.2 Tasa de fraude por tipo de tarjeta
    tasa_tarjeta = df.groupby('tipo_tarjeta')['fraude'].mean() * 100
    axes[0, 1].bar(tasa_tarjeta.index, tasa_tarjeta.values, color=['#3498db', '#e74c3c'])
    axes[0, 1].set_title('Tasa de Fraude por Tipo de Tarjeta', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Tasa de Fraude (%)')
    
    # 1.3 Transacciones por hora
    trans_hora = df.groupby('hora').size()
    fraude_hora = df[df['fraude']==1].groupby('hora').size()
    axes[1, 0].bar(trans_hora.index, trans_hora.values, alpha=0.6, label='Total', color='steelblue')
    if len(fraude_hora) > 0:
        axes[1, 0].bar(fraude_hora.index, fraude_hora.values, alpha=0.8, label='Fraudes', color='crimson')
    axes[1, 0].set_title('Transacciones por Hora', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Hora')
    axes[1, 0].set_ylabel('Cantidad')
    axes[1, 0].legend()
    
    # 1.4 Distribución por categoría
    cats = df['categoria'].value_counts()
    axes[1, 1].bar(cats.index, cats.values, color='teal')
    axes[1, 1].set_title('Transacciones por Categoría', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylabel('Cantidad')
    
    plt.tight_layout()
    plt.savefig('graficos/fig1_eda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figura 1: Análisis Exploratorio guardada")
    
    # Figura 2: Análisis de Fraude
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2.1 Distribución de fraudes
    fraude_counts = df['fraude'].value_counts()
    axes[0].pie(fraude_counts.values, labels=['Legítimas', 'Fraudes'], 
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], 
                explode=[0, 0.1], startangle=90)
    axes[0].set_title('Distribución de Transacciones', fontweight='bold')
    
    # 2.2 Monto promedio por fraude
    monto_prom = df.groupby('fraude')['monto'].mean()
    axes[1].bar(['Legítimas', 'Fraudes'], monto_prom.values, color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Monto Promedio', fontweight='bold')
    axes[1].set_ylabel('Monto ($)')
    
    # 2.3 Tasa de fraude por país
    tasa_pais = df.groupby('pais')['fraude'].mean() * 100
    axes[2].bar(tasa_pais.index, tasa_pais.values, color=['#3498db', '#e74c3c'])
    axes[2].set_title('Tasa de Fraude por Ubicación', fontweight='bold')
    axes[2].set_ylabel('Tasa de Fraude (%)')
    
    plt.tight_layout()
    plt.savefig('graficos/fig2_fraude.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figura 2: Análisis de Fraude guardada")
    
    # Figura 3: Correlaciones
    fig3, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ['monto', 'hora', 'dia_semana', 'numero_intentos', 
                    'distancia_promedio', 'transacciones_ultima_hora']
    corr = df[numeric_cols + ['fraude']].corr()
    
    im = ax.imshow(corr, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Correlación')
    ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graficos/fig3_correlaciones.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figura 3: Correlaciones guardada")
    
    print("\n✓ Análisis exploratorio completado")
    return df

if __name__ == "__main__":
    from download_dataset import descargar_dataset
    df = descargar_dataset()
    analisis_exploratorio(df)
