from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class AgentEvaluator:
    def __init__(self, pred_dir, gold_dir, lambda_weight=0.5, pred_filename="prediction.csv", gold_filename="gold.csv"):
        self.pred_dir = pred_dir
        self.gold_dir = gold_dir
        self.lambda_weight = lambda_weight
        self.pred_filename = pred_filename
        self.gold_filename = gold_filename

    @staticmethod
    def normalize_value(val):
        """Aplica las reglas de normalización del apartado 6.5"""
        if pd.isna(val):
            return ""
        
        val_str = str(val).strip()
        
        # 1. Valores Nulos (Null Values) -> ""[cite: 1]
        null_equivalents = {"", "null", "none", "nan", "nat", "<na>"}
        if val_str.lower() in null_equivalents:
            return ""
        
        # 2. Numéricos (Numeric) -> 2 decimales con ROUND_HALF_UP[cite: 1]
        try:
            float(val_str)
            d = Decimal(val_str)
            return str(d.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP))
        except Exception:
            pass
            
        # 3. Fechas y Datetime (Date / DateTime)[cite: 1]
        try:
            if '-' in val_str or '/' in val_str:
                dt = pd.to_datetime(val_str)
                if dt.time() == pd.Timestamp('00:00:00').time():
                    return dt.strftime('%Y-%m-%d')
                else:
                    if dt.tzinfo is not None:
                        return dt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                    else:
                        return dt.isoformat()
        except Exception:
            pass
            
        # 4. Cadenas de texto (String) -> Quitar \r\n, mantener mayúsculas/minúsculas[cite: 1]
        val_str = val_str.replace('\r', '').replace('\n', '')
        return val_str

    def get_column_signature(self, series):
        """Normaliza la columna y ordena sus celdas para construir la firma (6.2)"""
        normalized = series.apply(self.normalize_value)
        return tuple(normalized.sort_values(ignore_index=True).tolist())

    def check_name_field_combinations(self, pred_df, gold_sigs, pred_counts, gold_counts):
        """Soporte para Name Fields (6.5)"""
        missing_gold_sigs = []
        for sig, count in gold_counts.items():
            matched_count = min(count, pred_counts.get(sig, 0))
            if matched_count < count:
                missing_gold_sigs.extend([sig] * (count - matched_count))

        if not missing_gold_sigs:
            return 0
            
        extra_matches = 0
        cols = pred_df.columns.tolist()
        used_gold_sigs = set()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                combined_series_1 = pred_df[cols[i]].astype(str) + " " + pred_df[cols[j]].astype(str)
                combined_series_2 = pred_df[cols[j]].astype(str) + " " + pred_df[cols[i]].astype(str)
                
                sig1 = self.get_column_signature(combined_series_1)
                sig2 = self.get_column_signature(combined_series_2)
                
                for sig in (sig1, sig2):
                    if sig in missing_gold_sigs and sig not in used_gold_sigs:
                        extra_matches += 1
                        used_gold_sigs.add(sig)
                        missing_gold_sigs.remove(sig)
                        break

        return extra_matches

    @staticmethod
    def get_color_indicator(score):
        """Devuelve un emoji de color según el rango del score"""
        if score >= 0.9: return "🟢 (Excelente)"
        elif score >= 0.7: return "🟡 (Bueno)"
        elif score >= 0.4: return "🟠 (Regular)"
        else: return "🔴 (Deficiente)"

    def evaluate_task(self, task_name):
        """Evalúa una sola carpeta de tarea (task_<n>) devolviendo un score entre 0 y 1"""
        pred_task_dir = os.path.join(self.pred_dir, task_name)
        gold_task_dir = os.path.join(self.gold_dir, task_name)
        
        pred_path = os.path.join(pred_task_dir, self.pred_filename)
        gold_path = os.path.join(gold_task_dir, self.gold_filename)
        
        if not os.path.exists(gold_path):
            csvs = glob.glob(os.path.join(gold_task_dir, "*.csv"))
            if csvs: gold_path = csvs[0]
            else: return 0.0, {"status": "❌ ERROR", "message": "Falta archivo Gold"}

        if not os.path.exists(pred_path):
            csvs = glob.glob(os.path.join(pred_task_dir, "*.csv"))
            if csvs: pred_path = csvs[0]
            # Si prediction.csv no existe, el score es 0 según criterio 6.6[cite: 1]
            else: return 0.0, {"status": "⚠️ MISSING  ", "message": "Predicción ausente"}
            
        try:
            pred_df = pd.read_csv(pred_path, dtype=str, encoding='utf-8')
            gold_df = pd.read_csv(gold_path, dtype=str, encoding='utf-8')
        except Exception as e:
            return 0.0, {"status": "❌ ERROR", "message": f"Error I/O: {str(e)}"}
            
        gold_columns = len(gold_df.columns)
        predicted_columns = len(pred_df.columns)
        
        if gold_columns == 0:
            return (1.0 if predicted_columns == 0 else 0.0), {"status": "⚠️ EMPTY", "message": "Gold vacío"}

        pred_sigs = [self.get_column_signature(pred_df[c]) for c in pred_df.columns]
        gold_sigs = [self.get_column_signature(gold_df[c]) for c in gold_df.columns]
        
        pred_counts = Counter(pred_sigs)
        gold_counts = Counter(gold_sigs)
        
        matched_columns = 0
        for sig, g_count in gold_counts.items():
            if sig in pred_counts:
                matched_columns += min(g_count, pred_counts[sig])
                
        if matched_columns < gold_columns and predicted_columns > matched_columns:
            combined_matches = self.check_name_field_combinations(pred_df, gold_sigs, pred_counts, gold_counts)
            matched_columns += combined_matches
            
        extra_columns = max(0, predicted_columns - matched_columns)
        recall = matched_columns / gold_columns
        
        if predicted_columns > 0:
            score = recall - (self.lambda_weight * (extra_columns / predicted_columns))
        else:
            score = 0.0
            
        final_score = max(0.0, score) # Límite inferior 0[cite: 1]
        
        return final_score, {
            "status": "✅ OK",
            "gold_columns": gold_columns,
            "predicted_columns": predicted_columns,
            "matched_columns": matched_columns,
            "extra_columns": extra_columns,
            "recall": round(recall, 4),
            "score": round(final_score, 4)
        }

    def evaluate_all(self):
        """Busca todas las subcarpetas, las ordena de forma natural y genera el reporte visual limpio"""
        if not os.path.exists(self.gold_dir):
            print("❌ El directorio general 'gold' no existe.")
            return 0.0
            
        task_folders = [f for f in os.listdir(self.gold_dir) if os.path.isdir(os.path.join(self.gold_dir, f))]
        
        if not task_folders:
            print("❌ No se encontraron subcarpetas task_<n> dentro de gold.")
            return 0.0
            
        # Orden natural por número de tarea[cite: 1]
        try:
            task_folders_sorted = sorted(task_folders, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
        except Exception:
            task_folders_sorted = sorted(task_folders)

        total_score = 0.0
        
        print("\n" + "="*85)
        print(f" {'EVALUACIÓN DE AGENTE EN BATCH':^83} ")
        print("="*85)
        print(f"{'Tarea / Pregunta':<16} | {'Estado':<11} | {'Gold':<5} | {'Pred':<5} | {'Match':<5} | {'Score':<6} | {'Rendimiento'}")
        print("-"*85)

        for task_name in task_folders_sorted:
            score, details = self.evaluate_task(task_name)
            total_score += score
            
            if details["status"] == "✅ OK":
                color = self.get_color_indicator(details['score'])
                print(f"{task_name:<16} | {details['status']:<11} | {details['gold_columns']:^5} | {details['predicted_columns']:^5} | {details['matched_columns']:^5} | {details['score']:^6.2f} | {color}")
            else:
                msg = details.get("message", "Error")
                color = self.get_color_indicator(0.0)
                print(f"{task_name:<16} | {details['status']:<11} | {'-':^5} | {'-':^5} | {'-':^5} | {score:^6.2f} | {color} -> {msg}")

        final_average = total_score / len(task_folders_sorted)
        final_color = self.get_color_indicator(final_average)
        
        print("="*85)
        print(f" 📊 RESUMEN GENERAL:")
        print(f" • Total de Tareas Evaluadas : {len(task_folders_sorted)}")
        print(f" • Penalización Aplicada (λ)  : {self.lambda_weight}")
        print(f" • SCORE FINAL PROMEDIO      :  {final_average:.4f}   ({final_color})")
        print("="*85 + "\n")
        
        return final_average

if __name__ == '__main__':
    # Modifica estas rutas con tus carpetas principales
    DIR_PREDICCIONES = "C:\\Users\\laura.cristobal\\Desktop\\KDD\\KDD-modificado-feature_andrea_laura\\artifacts\\runs\\20260518T114202Z" 
    DIR_SOLUCIONES = "C:\\Users\\laura.cristobal\\Desktop\\KDD\\KDD-modificado-feature_andrea_laura\\data\\public\\output"

    evaluador = AgentEvaluator(pred_dir=DIR_PREDICCIONES, gold_dir=DIR_SOLUCIONES, lambda_weight=0.5)
    evaluador.evaluate_all()
    