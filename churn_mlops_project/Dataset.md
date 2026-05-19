# Documentación Dataset

## Introducción
Este dataset es de una empresa de telecomunicaciones y sirve para predecir qué clientes se van a dar de baja (lo que llamamos *churn*). Básicamente, analizamos sus datos para entender por qué se van.

## Descripción de las Variables
Para que el modelo funcione, dividimos la información en estos grupos sencillos:

*   **Datos Personales:** Información básica como el género, si son adultos mayores o si tienen pareja y dependientes.
*   **Servicios Contratados:** Aquí vemos qué tienen activado, como el tipo de internet (fibra o DSL), si tienen teléfono, seguridad online, soporte técnico o servicios de streaming.
*   **Detalles del Contrato:** Esto es clave. Incluye cuánto tiempo llevan en la empresa (tenure), qué tipo de contrato tienen (mes a mes o por años), cómo pagan y si usan factura electrónica.
*   **Costos:** Lo que pagan al mes y el total acumulado. En el código arreglamos los valores que faltaban en el total para que no den error.
*   **Churn (Variable Objetivo):** Es lo que queremos predecir. Si el cliente se fue (1) o se quedó (0).

## Preprocesamiento
En el script hicimos un par de cosas rápidas para limpiar todo:
1.  Quitamos el `customerID` porque no sirve para predecir nada.
2.  Convertimos los textos a números (como el género o si tienen pareja).
3.  Usamos `get_dummies` para que todas las categorías se vuelvan números y el modelo no falle.
4.  Separamos los datos en entrenamiento y prueba para ver qué tan bien aprende el modelo.

## Archivos Generados
Al final, el código guarda estos archivos en `data/processed/`:
*   `X_train.csv` y `y_train.csv`: Para que el modelo aprenda.
*   `X_test.csv` y `y_test.csv`: Para probar si el modelo de verdad funciona con datos que no conoce.
