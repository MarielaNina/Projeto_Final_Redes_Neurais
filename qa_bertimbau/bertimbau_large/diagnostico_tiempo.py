# ============================================================================
# DIAGNÓSTICO: ¿Por qué Training Time sale NaN?
# ============================================================================
# Agrega esta celda a tu notebook para diagnosticar el problema

import os

# Los experimentos que tienen NaN en tu screenshot
problematic_experiments = {
    "large QA - 2 epochs": "2_epochs_base_qa",  # o el nombre correcto
    "DoRA - 3 epochs (lr=2e-4)": "dora-epoch-3_lr_2e-4"
}

BASE_RESULTS = "/data/nina/qa_squad/qa_bertimbau/bertimbau_large/results"

print("="*80)
print("DIAGNÓSTICO: ¿Por qué no se encuentra el tiempo de entrenamiento?")
print("="*80)

for name, key in problematic_experiments.items():
    print(f"\n📊 Experimento: {name}")
    print(f"   Buscando: {key}")
    
    # Buscar experimento
    exp_path = None
    for d in os.listdir(BASE_RESULTS):
        if key.lower() in d.lower():
            exp_path = os.path.join(BASE_RESULTS, d)
            break
    
    if exp_path is None:
        print(f"   ❌ No se encontró la carpeta")
        continue
    
    print(f"   ✓ Carpeta encontrada: {os.path.basename(exp_path)}")
    
    # Buscar archivos tfevents
    print(f"\n   Buscando archivos tfevents...")
    event_files = []
    
    for root, dirs, files in os.walk(exp_path):
        for f in files:
            if "tfevents" in f:
                full_path = os.path.join(root, f)
                # Mostrar la ruta relativa para que sea más legible
                rel_path = full_path.replace(exp_path, "")
                event_files.append(rel_path)
                print(f"     ✓ Encontrado: {rel_path}")
    
    if not event_files:
        print(f"     ❌ NO se encontraron archivos tfevents")
        print(f"     💡 Este experimento no tiene logs de TensorBoard")
        
        # Listar qué hay en el experimento
        print(f"\n   Contenido de la carpeta:")
        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                print(f"     📁 {item}/")
            else:
                print(f"     📄 {item}")
    else:
        print(f"   ✓ Total encontrados: {len(event_files)} archivos")
        
        # Intentar leer con TensorBoard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_path = os.path.dirname(os.path.join(exp_path, event_files[0]))
        print(f"\n   Intentando leer eventos desde: {event_path}")
        
        try:
            ea = EventAccumulator(event_path)
            ea.Reload()
            
            tags = ea.Tags().get("scalars", [])
            print(f"   ✓ Tags encontrados: {len(tags)}")
            
            times = []
            for tag in tags:
                for e in ea.Scalars(tag):
                    times.append(e.wall_time)
            
            print(f"   ✓ Timestamps recopilados: {len(times)}")
            
            if len(times) < 2:
                print(f"   ⚠️  PROBLEMA: Se necesitan al menos 2 timestamps")
                print(f"   💡 El entrenamiento puede estar incompleto o no se loggeó correctamente")
            else:
                duration = max(times) - min(times)
                print(f"   ✓ Duración calculada: {duration/60:.1f} minutos")
                print(f"   ✓ Inicio: {min(times)}, Fin: {max(times)}")
                
        except Exception as e:
            print(f"   ❌ Error al leer TensorBoard: {e}")

print("\n" + "="*80)
print("POSIBLES SOLUCIONES:")
print("="*80)
print("""
1. Si NO hay archivos tfevents:
   - Esos experimentos no tienen logs de TensorBoard
   - Opción A: Usar training_time_from_files() en su lugar
   - Opción B: Dejar el tiempo como None/NaN
   
2. Si HAY archivos pero no se leen:
   - Verificar que estén en la ubicación correcta
   - Pueden estar en checkpoints/runs/ en lugar de la raíz

3. Si hay pocos timestamps:
   - El entrenamiento puede haber sido interrumpido
   - O no se configuró logging correctamente
""")
