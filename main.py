#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================
PROYECTO DE MINERÍA DE DATOS: Detección de Fraude Bancario
===============================================================
Pipeline Principal - Ejecuta todas las etapas en secuencia

Autor: Estudiante
Universidad Estatal Amazónica
Asignatura: Minería de Datos (UEA-L-UFPTI-009)
===============================================================
"""

from download_dataset import descargar_dataset
from eda import analisis_exploratorio
from preprocessing import preprocesar_datos
from modeling import entrenar_modelos
from evaluation import evaluar_modelos

def main():
    print("=" * 70)
    print("PIPELINE DE MINERÍA DE DATOS")
    print("Detección de Fraude en Transacciones Bancarias")
    print("=" * 70)
    
    # Etapa 1: Descarga de datos
    print("\n[ETAPA 1/5] Descargando dataset...")
    df = descargar_dataset()
    
    # Etapa 2: Análisis exploratorio
    print("\n[ETAPA 2/5] Realizando análisis exploratorio...")
    df_analizado = analisis_exploratorio(df)
    
    # Etapa 3: Preprocesamiento
    print("\n[ETAPA 3/5] Preprocesando datos...")
    df_procesado = preprocesar_datos(df_analizado)
    
    # Etapa 4: Modelado
    print("\n[ETAPA 4/5] Entrenando modelos...")
    modelos, X_train, X_test, y_train, y_test = entrenar_modelos(df_procesado)
    
    # Etapa 5: Evaluación
    print("\n[ETAPA 5/5] Evaluando modelos...")
    resultados = evaluar_modelos(modelos, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return resultados

if __name__ == "__main__":
    main()
