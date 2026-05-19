# 📉 Telco Customer Churn — Pipeline MLOps

> Pipeline de Machine Learning Operations (MLOps) de extremo a extremo para predecir el **abandono de clientes (Churn)** en una empresa de telecomunicaciones.

---

## 📋 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Requisitos Previos](#-requisitos-previos)
3. [Instalación](#-instalación)
4. [Estructura del Proyecto](#️-estructura-del-proyecto)
5. [Configuración (params.yaml)](#️-configuración-paramsyaml)
6. [Guía de Uso Paso a Paso](#-guía-de-uso-paso-a-paso)
   - [1. Entrenar el modelo](#1-entrenar-el-modelo-pipeline-completo)
   - [2. Ejecutar inferencia](#2-ejecutar-inferencia-producción)
   - [3. Correr pruebas unitarias](#3-correr-las-pruebas-unitarias-qa)
7. [Flujo del Pipeline](#-flujo-del-pipeline)
8. [Resultados del Mejor Modelo](#-resultados-del-mejor-modelo)
9. [Cómo Cambiar de Algoritmo](#-cómo-cambiar-de-algoritmo)
10. [Próximas Mejoras](#-próximas-mejoras)
11. [Contribución de LLM](#-contribución-de-llm-inteligencia-artificial)

---

## 📖 Descripción General

Este proyecto implementa un pipeline MLOps modular para clasificar si un cliente de telecomunicaciones **abandonará o no el servicio (Churn)**. El diseño separa claramente cada responsabilidad:

| Módulo | Responsabilidad |
|--------|----------------|
| `config/params.yaml` | Fuente única de verdad: rutas, hiperparámetros y configuración del modelo |
| `src/data_loader.py` | Ingesta, limpieza y transformación de datos crudos |
| `src/model_trainer.py` | Fábrica dinámica de modelos: entrena y serializa el algoritmo elegido |
| `src/main.py` | Orquestador que ejecuta el pipeline completo de punta a punta |
| `src/predict.py` | Script de inferencia para producción, con manejo robusto de errores |
| `tests/test_pipeline.py` | Suite de pruebas unitarias con `unittest` |

---

## 🧰 Requisitos Previos

- **Python** 3.9 o superior
- **pip** actualizado (`pip install --upgrade pip`)
- (Recomendado) Un entorno virtual para aislar las dependencias

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/DerJaguerPanzer/project-root.git
cd project-root/churn_mlops_project
```

### 2. Crear y activar un entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

Esto instalará automáticamente:
- `pandas` — manipulación de datos
- `numpy` — operaciones numéricas
- `scikit-learn` — modelos de ML (Random Forest, Logistic Regression, métricas)
- `joblib` — serialización del modelo `.pkl`
- `PyYAML` — lectura del archivo de configuración

---

## 🗂️ Estructura del Proyecto

```
project-root/
└── churn_mlops_project/
    ├── config/
    │   └── params.yaml          # ← Configuración central del pipeline
    ├── data/
    │   ├── raw/                 # Datos originales sin procesar
    │   └── processed/           # Datos limpios generados por data_loader.py
    ├── models/                  # Modelo entrenado (.pkl) generado por model_trainer.py
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py       # Carga, limpieza y feature engineering
    │   ├── model_trainer.py     # Entrenamiento y serialización del modelo
    │   ├── main.py              # Orquestador del pipeline completo
    │   └── predict.py           # Inferencia en producción
    ├── tests/
    │   └── test_pipeline.py     # Pruebas unitarias con unittest
    └── requirements.txt         # Dependencias del proyecto
```

---

## 🛠️ Configuración (params.yaml)

El archivo `config/params.yaml` es el **corazón del pipeline**. Controla todo el comportamiento sin tocar el código Python:

```yaml
data:
  raw_path: "data/raw/telco_churn.csv"
  processed_path: "data/processed/churn_clean.csv"
  test_size: 0.2
  random_state: 42
  target_column: "Churn"

model:
  algorithm: "LogisticRegression"   # Cambiar a "RandomForest" para alternar
  output_path: "models/churn_model.pkl"
  hyperparameters:
    max_iter: 1000
    C: 1.0
    random_state: 42
```

> **Tip:** Para cambiar de algoritmo, solo modifica el campo `algorithm` y los `hyperparameters` correspondientes. ¡No necesitas tocar ningún archivo `.py`!

---

## 🚀 Guía de Uso Paso a Paso

> Asegúrate de estar posicionado en `churn_mlops_project/` antes de ejecutar cualquier comando.

### 1. Entrenar el Modelo (Pipeline Completo)

Ejecuta la limpieza de datos y el entrenamiento del modelo definido en `params.yaml`:

```bash
python -m src.main
```

**¿Qué hace este comando?**
1. Lee `config/params.yaml` para obtener todas las rutas e hiperparámetros.
2. Llama a `data_loader.py`: carga el CSV crudo, imputa nulos con la mediana en `TotalCharges`, aplica mapeos binarios y One-Hot Encoding.
3. Guarda los datos limpios en `data/processed/churn_clean.csv`.
4. Llama a `model_trainer.py`: inicializa dinámicamente el algoritmo configurado, lo entrena con los datos procesados y lo serializa.
5. Guarda el modelo en `models/churn_model.pkl`.

**Salida esperada en consola:**
```
[INFO] Cargando datos desde: data/raw/telco_churn.csv
[INFO] Datos procesados guardados en: data/processed/churn_clean.csv
[INFO] Entrenando modelo: LogisticRegression
[INFO] Accuracy: 0.8211 | Recall: 0.6005 | F1: 0.6400
[INFO] Modelo guardado en: models/churn_model.pkl
[SUCCESS] Pipeline completado exitosamente.
```

---

### 2. Ejecutar Inferencia (Producción)

Una vez entrenado el modelo, puedes simular la predicción sobre un cliente nuevo:

```bash
python -m src.predict
```

**¿Qué hace este comando?**
- Verifica que el archivo `models/churn_model.pkl` exista (lanza un error descriptivo si no).
- Carga el modelo serializado.
- Construye un cliente de ejemplo con las mismas features que el modelo espera.
- Imprime la predicción: `0 = No abandona` / `1 = Abandona (Churn)`.

**Salida esperada:**
```
[INFO] Cargando modelo desde: models/churn_model.pkl
[INFO] Realizando predicción...
[RESULTADO] Predicción: 1 — El cliente tiene riesgo de abandonar (Churn).
```

> **Para producción real:** Modifica el diccionario de cliente en `src/predict.py` con los datos del cliente a evaluar, o adapta el script para recibir datos desde una API o base de datos.

---

### 3. Correr las Pruebas Unitarias (QA)

Para validar la integridad del pipeline antes de desplegar cambios:

```bash
python -m unittest tests/test_pipeline.py
```

**¿Qué se prueba?**
- ✅ Que los datos procesados no contienen valores nulos.
- ✅ Que las columnas esperadas existen tras el encoding.
- ✅ Que el modelo entrenado retorna métricas válidas (accuracy > 0.5).
- ✅ Que el archivo `.pkl` se genera correctamente.

**Salida esperada:**
```
....
----------------------------------------------------------------------
Ran 4 tests in 3.142s

OK
```

---

## 🔄 Flujo del Pipeline

```
[CSV Crudo]
     │
     ▼
data_loader.py
  ├─ Imputar nulos (mediana en TotalCharges)
  ├─ Mapeos binarios (Sí/No → 1/0)
  └─ One-Hot Encoding (columnas categóricas)
     │
     ▼
[CSV Limpio] ──► model_trainer.py
                  ├─ Lee algoritmo desde params.yaml
                  ├─ Train/Test Split (80/20)
                  ├─ Entrena modelo
                  └─ Serializa → churn_model.pkl
                       │
                       ▼
                  predict.py
                  └─ Inferencia sobre nuevos clientes
```

---

## 📊 Resultados del Mejor Modelo

Se evaluaron dos algoritmos sobre el dataset Telco Customer Churn:

| Algoritmo | Accuracy | Recall | F1-Score |
|-----------|----------|--------|----------|
| Random Forest | ~79% | ~52% | ~57% |
| **Logistic Regression** ✅ | **82.11%** | **60.05%** | **64.00%** |

La **Regresión Logística** fue el modelo ganador, priorizando el **Recall** (sensibilidad) ya que en un problema de Churn es más costoso **no detectar** a un cliente que sí va a abandonar.

---

## 🔀 Cómo Cambiar de Algoritmo

Para entrenar con **Random Forest** en lugar de Logistic Regression, edita `config/params.yaml`:

```yaml
model:
  algorithm: "RandomForest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

Luego vuelve a ejecutar:

```bash
python -m src.main
```

No se requiere ningún otro cambio en el código. ✅

---

## 🔭 Próximas Mejoras

- [ ] Aplicar `StandardScaler` a columnas numéricas para optimizar la convergencia de la Regresión Logística.
- [ ] Implementar `GridSearchCV` o `RandomizedSearchCV` para ajuste automático de hiperparámetros.
- [ ] Agregar manejo de clases desbalanceadas con `class_weight='balanced'` o técnicas SMOTE.
- [ ] Exponer el modelo como API REST con **FastAPI** o **Flask**.
- [ ] Integrar **MLflow** para tracking de experimentos.
- [ ] Añadir pipeline de CI/CD con **GitHub Actions** para ejecutar tests automáticamente.

---

## 🤖 Contribución de LLM (Inteligencia Artificial)

Este proyecto fue desarrollado y estructurado con la asistencia de un Modelo de Lenguaje Grande (LLM). Las principales contribuciones incluyeron:

1. **Diseño de Arquitectura MLOps:** Orientación para separar correctamente las responsabilidades (Data Loader, Model Trainer, Predictor) evitando scripts monolíticos.
2. **Gestión de Configuración (YAML):** Estructuración del `params.yaml` de forma dinámica, permitiendo cambiar algoritmos e hiperparámetros sin modificar código fuente.
3. **Resolución de Errores (Debugging):** Diagnóstico de `ModuleNotFoundError` (rutas relativas/absolutas), `KeyError` en lectura de YAML, y errores por hiperparámetros incompatibles en `scikit-learn`.
4. **Desarrollo de Pruebas y Producción (QA):** Redacción de pruebas unitarias con `unittest` y script de predicción robusto con manejo de excepciones.

---

<p align="center">
  Hecho con 🐍 Python · scikit-learn · PyYAML · joblib
</p>
