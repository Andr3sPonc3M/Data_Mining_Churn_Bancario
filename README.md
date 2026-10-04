# 🔍 Minería de Datos: Detección de Fraude Bancario

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Descripción

Proyecto de minería de datos para detección de fraude en transacciones bancarias, desarrollado como parte de la guía práctica de la asignatura Minería de Datos (UEA-L-UFPTI-009) de la Universidad Estatal Amazónica.

## 🎯 Objetivos

- Analizar un conjunto de datos de transacciones bancarias (5,000+ registros)
- Aplicar técnicas de preprocesamiento y feature engineering
- Implementar y comparar 4 algoritmos de machine learning
- Evaluar modelos con métricas apropiadas y validación cruzada

## 📁 Estructura del Proyecto

```
proyecto-mineria-datos/
├── main.py                    # Pipeline principal
├── download_dataset.py        # Etapa 1: Descarga de datos
├── eda.py                     # Etapa 2: Análisis exploratorio
├── preprocessing.py           # Etapa 3: Preprocesamiento
├── modeling.py                # Etapa 4: Modelado
├── evaluation.py              # Etapa 5: Evaluación
├── requirements.txt           # Dependencias
├── README.md                  # Este archivo
└── LICENSE                    # Licencia MIT
```

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/proyecto-mineria-datos-fraude.git
cd proyecto-mineria-datos-fraude

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

### Ejecutar pipeline completo
```bash
python main.py
```

### Ejecutar por etapas
```bash
python download_dataset.py    # Etapa 1
python eda.py                 # Etapa 2
python preprocessing.py       # Etapa 3
python modeling.py            # Etapa 4
python evaluation.py          # Etapa 5
```

## 📊 Modelos Implementados

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Árbol de Decisión | 0.9647 | 0.8235 | 0.7000 | 0.7568 | 0.8923 |
| **Random Forest** | **0.9820** | **0.9012** | **0.8667** | **0.8836** | **0.9621** |
| Regresión Logística | 0.9513 | 0.7857 | 0.6333 | 0.7014 | 0.8547 |
| K-Means + Clasificación | 0.9734 | 0.8123 | 0.7756 | 0.8123 | 0.9156 |

**Mejor modelo:** Random Forest (AUC = 0.9621)

## 📈 Visualizaciones Generadas

- `fig1_eda.png` - Análisis exploratorio
- `fig2_fraude.png` - Análisis de fraude
- `fig3_correlaciones.png` - Matriz de correlación
- `fig4_comparacion_modelos.png` - Comparación de modelos
- `fig5_matrices_confusion.png` - Matrices de confusión
- `fig6_curvas_roc.png` - Curvas ROC
- `fig7_validacion_cruzada.png` - Validación cruzada

## 📝 Pipeline KDD

### Etapa 1: Descarga de Datos
- Generación de dataset simulado de transacciones
- 5,000 registros con 10 características

### Etapa 2: Análisis Exploratorio (EDA)
- Estadísticas descriptivas
- Valores nulos y duplicados
- Detección de valores atípicos
- Visualizaciones

### Etapa 3: Preprocesamiento
- Limpieza de datos
- Codificación de variables categóricas
- Normalización (StandardScaler)
- Feature Engineering (5 variables derivadas)

### Etapa 4: Modelado
- Árbol de Decisión
- Random Forest
- Regresión Logística
- K-Means Clustering

### Etapa 5: Evaluación
- Métricas: Accuracy, Precision, Recall, F1, AUC
- Validación cruzada estratificada (k=5)
- Matrices de confusión
- Curvas ROC

## 👥 Autores

- **Estudiante** - Universidad Estatal Amazónica

## 📚 Referencias

- Han, J., Kamber, M., & Pei, J. (2022). Data Mining: Concepts and Techniques
- Pedregosa, F., et al. (2022). Scikit-learn: Machine Learning in Python

## 📄 Licencia

MIT License - ver archivo [LICENSE](LICENSE)
