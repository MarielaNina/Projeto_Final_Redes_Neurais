# qa_squad_bertimbau

Proyecto para entrenamiento y evaluación de modelos BERT en tareas de Question Answering (SQuAD) en portugués.

Descripción
- Contiene código para entrenar variantes base y large de modelos BERT adaptados a portugués (Bertimbau), con configuraciones para LoRA/QLoRA.
- Incluye scripts de preprocesamiento, postprocesamiento, y varios `main_*.py` para distintos flujos de entrenamiento y predicción.

Estructura principal
- `qa_bertimbau/` : subproyecto con implementaciones para `bertimbau_base`, `bertimbau_large` y `tucano_base`.
  - `bertimbau_base/` y `bertimbau_large/`: scripts para entrenamiento (`main.py`, `main_qlora.py`, `main_lora.py`, etc.), junto con `data/` y `results/`.
- `data/`: ubicación esperada para archivos JSON de entrenamiento/validación (si no están, los scripts intentan generar archivos "flattened").

Cómo empezar (rápido)
1. Crear y activar un entorno virtual (recomendado con conda):
	```sh
	python -m venv .venv
	.venv/bin/activate
	```
	o con conda:
	```sh
	conda create -n qa_squad python=3.10 -y
	conda activate qa_squad
	```
2. Instalar dependencias (puedes revisar `qa_bertimbau/requirements.txt`):
	```sh
	pip install -r qa_bertimbau/requirements.txt
	```
3. Ejecutar un ejemplo de entrenamiento (ajusta batch sizes y nombres según tu GPU):
	```sh
	cd qa_bertimbau/bertimbau_base
	python main_qlora.py
	```

Notas importantes
- No se suben artefactos pesados (modelos, checkpoints) gracias a `.gitignore`. Evita commitear archivos grandes como `*.pt`, `*.safetensors`, `runs/`, `results/`.
- Si quieres subir un modelo o checkpoint, considera usar un servicio de almacenamiento (Hugging Face Hub, Google Drive, S3) en vez del repositorio Git.
- Hay una conversión reciente del subproyecto `qa_bertimbau` a carpeta normal dentro del repo principal. El historial independiente de ese subproyecto NO se preservó en este repositorio; si necesitas el historial, lo podemos reconstituir en un repo separado.

Contribuir
- Abre un issue o pull request en GitHub.
- Antes de commitear, ejecuta linters y pruebas (si las añades).

Contacto
- Maintainer: `MarielaNina` (https://github.com/MarielaNina)

Licencia
- Añade una licencia si lo deseas (p. ej. MIT). Puedo añadir un `LICENSE` por ti si me indicas cuál prefieres.
