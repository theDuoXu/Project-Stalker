---
title: "Sistema de Soporte a la Decisión para Calidad del Agua: Gemelo Digital HPC en Tiempo Real y Grafo de Conocimiento Hidrológico"
author: "Duo Xu Ye"
date: "Enero 2026"
project_id: "7942c954-ea70-4cd8-b51f-501011cf2ddb"
---

# 1. Introducción y Definición del Alcance

## 1.1. Contexto del Problema
La gestión de la calidad del agua en grandes cuencas hidrográficas enfrenta un desafío crítico: la detección temprana y localización precisa de eventos de contaminación. En la actualidad, organismos como la Confederación Hidrográfica del Tajo (CHT) operan redes de monitoreo extensas, pero la identificación del origen de un vertido sigue siendo un proceso reactivo y técnicamente complejo.

El problema central radica en la naturaleza física del fenómeno: la identificación de vertidos es un problema inverso mal condicionado (ill-posed). Cuando una estación de control detecta una anomalía en parámetros físico-químicos (como amonio, nitratos o pH) , el contaminante ya ha sufrido procesos de transporte, difusión y mezcla a lo largo del cauce durante un tiempo indeterminado.

En el caso específico de la cuenca del Tajo, la dimensionalidad del problema hace inviable una resolución analítica simple. La red de vigilancia dispone de aproximadamente 32 estaciones del Sistema Automático de Información de Calidad de las Aguas (SAICA) para monitorizar un dominio que contiene más de 1.800 puntos de vertido autorizados, además de un número desconocido de focos de contaminación difusa o ilegal. Esta desproporción entre puntos de observación (sensores) y posibles fuentes emisoras convierte la inferencia del origen en un reto computacional que supera las capacidades de los sistemas de vigilancia tradicionales.

La incapacidad para determinar la fuente en tiempo real tiene consecuencias operativas y ambientales directas. La detección tardía impide la contención inmediata del vertido, incrementa los costes de remediación y dificulta la imputación de responsabilidades legales. Ante este escenario, se hace necesario un cambio de paradigma: transitar de una vigilancia pasiva a un Sistema de Soporte a la Decisión (DSS) proactivo , capaz de integrar simulación física de alta fidelidad con datos de campo para reducir la incertidumbre sobre el origen del contaminante
### 1.1.1. La Monitorización de Vertidos en Cuencas Hidrográficas
La vigilancia de la calidad del agua en España se articula principalmente a través del Sistema Automático de Información de Calidad de las Aguas (SAICA). En la demarcación hidrográfica del Tajo, esta infraestructura se complementa con el Sistema Automático de Información Hidrológica (SAIH), encargado de medir caudales y niveles.

Sin embargo, el paradigma actual de monitorización presenta limitaciones estructurales severas para la detección de vertidos:

1. Escasez Espacial de Puntos de Control: La red SAICA en la zona de estudio dispone de aproximadamente 32 estaciones de medición en tiempo real. Estas estaciones deben fiscalizar un dominio hidrológico que recibe presiones de más de 1.800 puntos de vertido autorizados (según el Censo de Vertidos de la CHT), sin contar las fuentes difusas o ilegales. Esta relación de ~1:60 (sensor por vertido) implica que grandes tramos del río permanecen "ciegos" a la vigilancia continua, dependiendo exclusivamente de la dispersión de contaminantes aguas abajo para su detección.
2. Naturaleza de los Parámetros Monitorizados: Los sensores despliegan mediciones de parámetros físico-químicos generales: Temperatura, pH, Conductividad, Oxígeno Disuelto, Turbidez, Amonio y Nitratos. Aunque eficaces para evaluar la salud general del ecosistema, estos indicadores carecen de una "huella digital" específica que vincule unívocamente una anomalía con una industria o fuente concreta. Por ejemplo, un pico de amonio puede provenir tanto de una Estación Depuradora de Aguas Residuales (EDAR) urbana como de una escorrentía agrícola.
3. Latencia en la Cadena de Información: Actualmente, la detección de eventos es fundamentalmente reactiva. Los datos se transmiten y almacenan, pero el análisis de anomalías a menudo ocurre a posteriori o mediante umbrales estáticos simples. No existe un sistema operativo que integre la hidrodinámica del río (velocidad y caudal del SAIH) para contextualizar dinámicamente si una lectura anómala en un punto B corresponde a un vertido ocurrido horas antes en un punto A aguas arriba.

Esta brecha entre la capacidad de recolección de datos (limitada y dispersa) y la realidad física del río (continua y dinámica) es lo que motiva el desarrollo de una capa de software intermedia: un Gemelo Digital capaz de rellenar los vacíos de información mediante simulación física acelerada.
### 1.1.2. Limitaciones de los Sistemas de Simulación Tradicionales

Aunque la modelización hidráulica es una disciplina madura, las herramientas convencionales de ingeniería civil (como HEC-RAS, Iber o MIKE) están diseñadas principalmente para estudios de planificación *offline*, como la delimitación de zonas inundables o el diseño de infraestructuras. Estos sistemas presentan limitaciones fundamentales cuando se intentan aplicar a la gestión operativa de emergencias en tiempo real:

1.  **Cuello de Botella Computacional (CPU-Bound):**
    La resolución numérica de las Ecuaciones de Aguas Someras (Shallow Water Equations) y del transporte de escalares requiere mallas computacionales finas para mantener la estabilidad numérica (condición CFL). Los motores de simulación tradicionales suelen ejecutarse en CPU, lo que limita drásticamente la velocidad de procesamiento. Simular horas de flujo físico puede tomar un tiempo equivalente o superior en tiempo de reloj ("tiempo real" o más lento), lo cual es inaceptable cuando se requiere una predicción inmediata tras una alerta.

2.  **Inviabilidad para la Resolución del Problema Inverso:**
    Dado que el origen del vertido es desconocido, localizarlo implica un enfoque iterativo o probabilístico: se deben simular miles de escenarios potenciales (variando ubicación, masa y duración) para encontrar cuál coincide con las lecturas observadas en los sensores.
    Con un *solver* tradicional basado en CPU, generar un banco de 100.000 escenarios para entrenar una IA o realizar una búsqueda de fuerza bruta requeriría meses de cómputo. Esta barrera de rendimiento hace imposible la inferencia de fuentes en tiempos operativos razonables sin el uso de aceleración masiva por hardware (HPC).

3.  **Desacoplamiento de los Datos en Tiempo Real:**
    La mayoría de los simuladores comerciales operan como silos aislados: requieren la preparación manual de ficheros de entrada y no están diseñados para ingerir flujos de datos de telemetría (IoT) en vivo. La falta de una arquitectura orientada a servicios impide que el modelo se actualice automáticamente con el estado hidrológico real (caudales del SAIH) en el momento del incidente, reduciendo la precisión de cualquier predicción de transporte.

En conclusión, la brecha entre la necesidad operativa (respuesta en minutos, miles de escenarios) y la capacidad tecnológica convencional (simulación lenta, escenario único) justifica la necesidad crítica de desarrollar un motor físico nativo acelerado por GPU.
## 1.2. Objetivos del Semestre (Fase 1: Infraestructura y Datos)

Dada la magnitud y complejidad del desafío descrito, un problema inverso mal condicionado en un dominio de alta dimensionalidad, el proyecto se ha estructurado en una ejecución bifásica. La presente memoria recoge los resultados de la **Fase 1**, cuyo propósito fundamental ha sido la construcción de la infraestructura tecnológica habilitadora.

El objetivo central de este semestre no ha sido resolver el problema inverso de forma inmediata, sino **desarrollar la capacidad computacional y analítica** necesaria para abordarlo. La premisa rectora es que ningún modelo de Inteligencia Artificial (DeepONet) puede operar eficazmente sin dos pre-requisitos funcionales: (1) un generador de datos sintéticos masivos de alta fidelidad física y (2) un flujo de datos de campo depurado y confiable.

En consecuencia, los esfuerzos de este periodo se han alineado para entregar una plataforma operativa *end-to-end*, centrada en tres pilares estratégicos: la simulación de alto rendimiento, la ingeniería de datos y la arquitectura de sistemas segura.
### 1.2.1. Desarrollo del Gemelo Digital de Alto Rendimiento (HPC)
### 1.2.2. Ingeniería de Datos: Pipeline de Limpieza y Grafo de Conocimiento
### 1.2.3. Implementación de Arquitectura de Software Híbrida y Segura

## 1.3. Estructura de la Memoria y Justificación Tecnológica

# 2. Fundamentos Teóricos y Tecnológicos

## 2.1. Modelado Físico de Sistemas Fluviales
### 2.1.1. Hidrodinámica: Ecuaciones de Saint-Venant y Aproximación de Manning
### 2.1.2. Transporte de Contaminantes: Ecuación de Advección-Difusión-Reacción

## 2.2. Computación Heterogénea y GPGPU
### 2.2.1. Arquitectura NVIDIA CUDA y Modelo de Ejecución SIMT
### 2.2.2. Métricas Críticas de Rendimiento: Occupancy y Speed-of-Light (SOL)

## 2.3. Calidad de Datos en Series Temporales Ambientales
### 2.3.1. Tipología de Anomalías: Flatlines, Outliers y Rupturas de Batch
### 2.3.2. Topología de Grafos para Modelado de Sistemas Hidrológicos

# 3. Arquitectura del Sistema de Software

## 3.1. Visión Global: Arquitectura Hexagonal y Políglota

## 3.2. Capa de Computación Nativa (Compute Engine)
### 3.2.1. Integración JNI (Java Native Interface) para Baja Latencia
### 3.2.2. Gestión de Ciclo de Vida de Librerías Dinámicas (.so / .dll)

## 3.3. Capa de Datos y Servicios (Data Engine)
### 3.3.1. Pipeline de Ingesta y Orquestación Dockerizada
### 3.3.2. Estrategia de Persistencia Híbrida (Cassandra, Neo4j, Parquet)
### 3.3.3. Seguridad: Proveedor de Identidad Keycloak (Standard Flow + PKCE)

## 3.4. Capa de Presentación (Desktop UI)
### 3.4.1. Arquitectura Reactiva con JavaFX y OpenJFX 25
### 3.4.2. Patrones de Comunicación Asíncrona con el Backend

# 4. Implementación del Motor Físico HPC (C++/CUDA)

## 4.1. Diseño del Solver Hidrodinámico
### 4.1.1. Discretización Espacial y Estructuras de Datos Alineadas en GPU
### 4.1.2. Kernel de Manning: Estrategias Branchless y Aritmética FP32

## 4.2. Optimización de Rendimiento y Memoria
### 4.2.1. Gestión de Memoria Pinned y Transferencias Asíncronas (DMA)
### 4.2.2. Coalescencia de Memoria y Optimización de Caché L1/L2

## 4.3. Validación de Eficiencia Computacional
### 4.3.1. Análisis de Profiling con NVIDIA Nsight Compute
### 4.3.2. Logro del 90% de Compute Speed-of-Light (SOL)
### 4.3.3. Benchmark Comparativo: CPU (Ryzen 9 5900x) vs GPU (RTX 5090)

# 5. Ingeniería de Datos y Pipeline de Limpieza

## 5.1. Ingesta y Normalización de Fuentes Heterogéneas
### 5.1.1. Scraping y Unificación de Series Temporales (SAICA/SAIH)
### 5.1.2. Tratamiento de Registros Administrativos (Censo de Vertidos)

## 5.2. Pipeline de Calidad y Detección de Anomalías
### 5.2.1. Detección Multinivel de Flatlines y Pérdida de Dinámica
### 5.2.2. Validación Física (Hard Limits) y Coherencia Temporal (Jump Limits)
### 5.2.3. Auditoría Forense de Datos: Ley de Benford y Análisis del Dígito Terminal

## 5.3. Construcción del Grafo de Conocimiento (Neo4j)
### 5.3.1. Modelado de Relaciones Espaciales y Topológicas (Vertido-Río-Sensor)
### 5.3.2. Algoritmo de Inferencia de Origen "Aguas Arriba"

# 6. Desarrollo de la Aplicación de Escritorio y Visualización

## 6.1. Configuración y Parametrización del Gemelo Digital
### 6.1.1. Editor de Geometría Fluvial y Propiedades Físicas (`RiverConfig`)
### 6.1.2. Gestión de Escenarios de Simulación ("Flight Recorder")

## 6.2. Visualización en Tiempo Real
### 6.2.1. Renderizado de la Mancha Contaminante y Dinámica de Fluidos
### 6.2.2. Telemetría en Vivo: Monitorización de FPS y Uso de VRAM

# 7. Resultados y Discusión

## 7.1. Evaluación del Rendimiento del Simulador
### 7.1.1. Escalabilidad del Solver ante Aumento de Celdas/Nodos
### 7.1.2. Impacto de la Aceleración GPU en la Experiencia de Usuario (Latencia)

## 7.2. Evaluación de la Calidad de Datos
### 7.2.1. Métricas de Recuperación de Datos e Imputación
### 7.2.2. Hallazgos en el Censo: Concentración (Curvas Lorenz/Gini) y Anomalías

## 7.3. Limitaciones Actuales y Desviaciones del Planteamiento Inicial

# 8. Conclusiones y Trabajo Futuro

## 8.1. Conclusiones del Semestre
## 8.2. Líneas de Trabajo Futuro (Semestre 2)
### 8.2.1. Generación Masiva de Dataset Sintético usando el Solver HPC Validado
### 8.2.2. Entrenamiento e Integración del Modelo DeepONet (IA Física)

# 9. Bibliografía y Referencias

# 10. Anexos
## 10.1. Reportes de Validación de NVIDIA Nsight Compute
## 10.2. Diagramas de Clases y Arquitectura Detallada
## 10.3. Capturas del Análisis Forense de Datos