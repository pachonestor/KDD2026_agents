from __future__ import annotations
 
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
 
# --- Configuración del Sistema de Evaluación (Sección 6) ---
LAMBDA_WEIGHT = 0  # Peso de la penalización por columnas redundantes
 
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
 
# Intentar importar dependencias del baseline
try:
    from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
    from data_agent_baseline.config import load_app_config
    from data_agent_baseline.run.runner import create_run_output_dir, run_single_task
except ImportError:
    print("Error: No se pudieron encontrar los módulos de 'data_agent_baseline'.")
    print("Asegúrate de ejecutar este script desde la raíz del proyecto.")
    sys.exit(1)
 
def normalize_cell(value: str | None) -> str:
    """Normaliza el contenido de la celda: string, sin espacios (Regla 6.2)."""
    if value is None:
        return ""
    return str(value).strip()
 
def get_column_signatures(path: Path) -> list[tuple[str, ...]]:
    """
    Lee el CSV y genera una firma para cada columna (Regla 6.2).
    - Ignora nombres de columnas (headers).
    - Ignora orden de filas mediante sorting.
    """
    if not path.exists():
        return []
 
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except Exception as e:
        print(f"Error leyendo {path.name}: {e}")
        return []
   
    if len(rows) <= 1: # Solo header o vacío
        return []
 
    # Ignorar la primera fila (header)
    data_rows = rows[1:]
   
    # Determinar el número de columnas basado en la fila más larga para evitar errores
    num_cols = max(len(row) for row in data_rows)
   
    # Inicializar contenedores de columnas
    columns = [[] for _ in range(num_cols)]
   
    # Transponer filas a columnas
    for row in data_rows:
        for i in range(num_cols):
            val = row[i] if i < len(row) else ""
            columns[i].append(normalize_cell(val))
 
    # Generar firmas: Tuplas de valores ordenados (Regla: "Ignore row order")
    signatures = [tuple(sorted(col)) for col in columns]
    return signatures
 
def calculate_score(pred_path: Path, gold_path: Path, lambd: float = LAMBDA_WEIGHT) -> float:
    """
    Calcula la métrica final según la Sección 6.3.
    Score = Recall - λ * (Extra Columns / Predicted Columns)
    """
    pred_sigs = get_column_signatures(pred_path)
    gold_sigs = get_column_signatures(gold_path)
 
    if not gold_sigs:
        return 0.0
    if not pred_sigs:
        return 0.0
 
    pred_counts = Counter(pred_sigs)
    gold_counts = Counter(gold_sigs)
 
    # Matched Columns: Columnas en prediction que existen en gold (considerando duplicados)
    matched_columns = 0
    for sig, count in gold_counts.items():
        if sig in pred_counts:
            # Match basado en la cantidad mínima presente en ambos (Regla 6.2 duplicados)
            matched_columns += min(count, pred_counts[sig])
 
    # Variables de la fórmula 6.3
    num_gold_cols = len(gold_sigs)
    num_pred_cols = len(pred_sigs)
    extra_cols = max(0, num_pred_cols - matched_columns)
 
    # Recall
    recall = matched_columns / num_gold_cols
   
    # Penalización
    penalty = lambd * (extra_cols / num_pred_cols)
   
    final_score = recall - penalty
   
    # El límite inferior es 0 (Regla 6.3)
    return max(0.0, final_score)
 
def infer_gold_root(dataset_root: Path) -> Path:
    if dataset_root.name == "input" and dataset_root.parent.name == "public":
        return dataset_root.parent / "output"
    return dataset_root.parent / "output"
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación automática basada en Column Matching (Sección 6).")
    parser.add_argument("--config", type=Path, required=True, help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--gold-root", type=Path, default=None, help="Ruta opcional a los gold.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Máximo de tareas a evaluar.")
   
    args = parser.parse_args()
 
    # Cargar configuración y dataset
    config = load_app_config(args.config)
    dataset = DABenchPublicDataset(config.dataset.root_path)
   
    if not dataset.exists:
        print(f"Error: Dataset root no encontrado en {config.dataset.root_path}")
        return 1
 
    gold_root = args.gold_root or infer_gold_root(config.dataset.root_path)
    if not gold_root.exists():
        print(f"Error: Carpeta Gold no encontrada en {gold_root}")
        return 1
 
    run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id="column_eval")
   
    print(f"--- Iniciando Evaluación Automática ---")
    print(f"Output Dir: {run_output_dir}")
    print(f"Gold Root:  {gold_root}")
    print(f"Lambda (λ): {LAMBDA_WEIGHT}")
    print("-" * 40)
 
    task_ids = dataset.list_task_ids()
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
 
    scores = []
    summary = {"failed": 0, "no_answer": 0, "gold_missing": 0}
 
    for task_id in task_ids:
        try:
            # Ejecutar la tarea del agente
            artifacts = run_single_task(task_id=task_id, config=config, run_output_dir=run_output_dir)
           
            if artifacts.prediction_csv_path is None:
                print(f"[{task_id}] Sin respuesta (CSV no generado).")
                print("-" * 40)
                scores.append(0.0)
                summary["no_answer"] += 1
                continue
 
            gold_path = gold_root / task_id / "gold.csv"
            if not gold_path.exists():
                print(f"[{task_id}] Gold missing en {gold_path}")
                print("-" * 40)
                summary["gold_missing"] += 1
                continue
 
            # Calcular Score con el nuevo método
            task_score = calculate_score(artifacts.prediction_csv_path, gold_path)
            scores.append(task_score)
            print(f"[{task_id}] Score: {task_score:.4f}")
            print("-" * 40)
 
        except Exception as exc:
            print(f"[{task_id}] Error crítico: {exc}")
            print("-" * 40)
            summary["failed"] += 1
 
    # Resumen Final
    print("\n" + "="*20 + " RESUMEN " + "="*20)
    if scores:
        avg_score = sum(scores) / len(task_ids)
        print(f"Puntaje Promedio (Total Tasks): {avg_score:.4f}")
   
    print(f"Tareas procesadas: {len(scores)}")
    print(f"Fallos técnicos:   {summary['failed']}")
    print(f"Sin predicción:    {summary['no_answer']}")
    print(f"Gold perdidos:     {summary['gold_missing']}")
    print("="*49)
 
    return 0
 
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEvaluación cancelada por el usuario.")
        sys.exit(1)