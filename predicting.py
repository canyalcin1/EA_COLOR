# predictor_ensemble.py
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
import torch

# Slim eğitim dosyasından yardımcıları içe aktar
from TrainRS400SLIMKFold import (
    load_artifacts, ensure_columns, coerce_numeric,
    INPUT_COLS, TARGET_COLS, ANGLES
)

# --- Yardımcılar ---

def row_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Satır-bazlı normalize: her satırın toplamı 1 olacak şekilde böl."""
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = np.nan
    Xn = X / (s + eps)
    Xn = np.nan_to_num(Xn, copy=False)
    return Xn

def cie76_delta_e_np(Lab_true: np.ndarray, Lab_pred: np.ndarray) -> np.ndarray:
    diff = Lab_true - Lab_pred
    return np.sqrt((diff ** 2).sum(axis=1))

def angle_triplet_available(target_cols, ang: str) -> bool:
    return (f"{ang}L" in target_cols) and (f"{ang}a" in target_cols) and (f"{ang}b" in target_cols)

def extract_triplet_indices(target_cols, ang: str):
    iL = target_cols.index(f"{ang}L")
    ia = target_cols.index(f"{ang}a")
    ib = target_cols.index(f"{ang}b")
    return iL, ia, ib

def safe_get(df: pd.DataFrame, col: str):
    return df[col].to_numpy(dtype=float) if col in df.columns else np.full((len(df),), np.nan, dtype=float)

def build_report_df(df_in: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    target_cols, angles_use) -> pd.DataFrame:
    """
    Çıktı yerleşimi:
    SampleNo/Row, DE15, DE25, ..., DE_mean,
    15L_real,15L_pred,15a_real,15a_pred,15b_real,15b_pred, ... (tüm açılar)
    15Si_real,15Si_pred,45Si_real,45Si_pred,75Si_real,75Si_pred,
    15Sa_real,15Sa_pred,45Sa_real,45Sa_pred,75Sa_real,75Sa_pred,
    G_real,G_pred (varsa)
    """
    n = len(df_in)
    out_cols = []
    data = {}

    # Kimlik kolonu
    if "SampleNo" in df_in.columns:
        data["SampleNo"] = df_in["SampleNo"].astype(str).fillna("").to_list()
        out_cols.append("SampleNo")
    else:
        data["Row"] = list(range(n))
        out_cols.append("Row")

    # ΔE kolonları
    de_cols = []
    for ang in angles_use:
        iL, ia, ib = extract_triplet_indices(target_cols, ang)
        true_trip = y_true[:, [iL, ia, ib]]
        pred_trip = y_pred[:, [iL, ia, ib]]
        mask_valid = ~np.any(np.isnan(true_trip), axis=1)
        de = np.full((n,), np.nan, dtype=float)
        if np.any(mask_valid):
            de[mask_valid] = cie76_delta_e_np(true_trip[mask_valid], pred_trip[mask_valid])
        col_name = f"DE{ang}"
        data[col_name] = de
        out_cols.append(col_name)
        de_cols.append(col_name)

    # ΔE_mean (satır bazlı)
    if de_cols:
        de_mat = np.vstack([data[c] for c in de_cols]).T
        data["DE_mean"] = np.nanmean(de_mat, axis=1)
        out_cols.append("DE_mean")

    # L/a/b gerçek ve tahmin kolonları
    for ang in angles_use:
        for comp in ("L", "a", "b"):
            name_true = f"{ang}{comp}_real"
            name_pred = f"{ang}{comp}_pred"
            idx = target_cols.index(f"{ang}{comp}")
            data[name_true] = y_true[:, idx]
            data[name_pred] = y_pred[:, idx]
            out_cols += [name_true, name_pred]

    # Si / Sa (varsa)
    for comp in ("Si", "Sa"):
        for ang in ("15", "45", "75"):
            name = f"{ang}{comp}"
            if name in target_cols:
                idx = target_cols.index(name)
                data[f"{name}_real"] = y_true[:, idx]
                data[f"{name}_pred"] = y_pred[:, idx]
                out_cols += [f"{name}_real", f"{name}_pred"]

    # G (varsa)
    if "G" in target_cols:
        idx = target_cols.index("G")
        data["G_real"] = y_true[:, idx]
        data["G_pred"] = y_pred[:, idx]
        out_cols += ["G_real", "G_pred"]

    return pd.DataFrame(data, columns=out_cols)

def summarize_to_console(report_df: pd.DataFrame, angles_use):
    lines = []
    for ang in angles_use:
        col = f"DE{ang}"
        if col in report_df.columns:
            v = report_df[col].astype(float)
            lines.append((col, np.nanmean(v), np.nanmedian(v)))
    if lines:
        print("🔎 Dataset ΔE özet (nan hariç):")
        for name, mean_v, med_v in lines:
            print(f"  {name}: mean={mean_v:.3f} | median={med_v:.3f}")
        if "DE_mean" in report_df.columns:
            dm = report_df["DE_mean"].astype(float)
            print(f"  DE_mean: mean={np.nanmean(dm):.3f} | median={np.nanmedian(dm):.3f}")

def collect_model_paths(single_model: str, many_models: list, model_dir: str, manifest_path: str) -> list:
    paths = []
    if many_models: paths.extend(many_models)
    if model_dir:
        paths.extend(sorted(glob.glob(os.path.join(model_dir, "*.pt"))))
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            js = json.load(f)
            paths.extend(js.get("models", []))
    if single_model:
        paths.append(single_model)
    # temizle
    uniq = []
    seen = set()
    for p in paths:
        if p and os.path.exists(p) and p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

# --- Ana akış ---

def run(models: list, csv_path: str, out_csv: str, angles_arg: str = None, strict_cols: bool = True):
    """
    models: 1+ model yolu (ensemble için birden fazla).
    strict_cols: True ise tüm modellerin input/target kolonları birebir aynı olmalı.
    """
    assert len(models) > 0, "Model listesi boş."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # İlk modelden kolon, scaler ve hedef bilgisini al
    model0, xsc0, ysc0, in_cols0, tgt_cols0, meta0 = load_artifacts(models[0], device)
    model0.eval()

    # Açı listesi
    if angles_arg:
        req_angles = [a.strip() for a in angles_arg.split(",") if a.strip()]
    else:
        req_angles = list(ANGLES)

    # Hangi açılar mevcut hedeflerle uyumlu?
    angles_use = [a for a in req_angles if angle_triplet_available(tgt_cols0, a)]
    if len(angles_use) == 0:
        raise SystemExit(f"Seçilen açılar hedeflerde yok. İstenen: {req_angles}, mevcut hedefler: {tgt_cols0}")
    missing = [a for a in req_angles if a not in angles_use]
    if missing:
        print(f"⚠️ Bu açılar hedeflerinizde yok, atlanacak: {','.join(missing)}")

    # CSV yükle + sayısal hale getir
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = ensure_columns(df, in_cols0 + tgt_cols0, fill=np.nan)
    df = coerce_numeric(df, in_cols0 + tgt_cols0)

    # X: pigmentler -> normalize -> ilk modelin scaler'ı (X)
    X_raw = df[in_cols0].to_numpy(dtype=np.float32)
    X_norm = row_normalize_np(X_raw)

    # Ensemble tahminleri
    preds = []
    for mpath in models:
        model_i, xsc_i, ysc_i, in_cols_i, tgt_cols_i, meta_i = load_artifacts(mpath, device)
        model_i.eval()

        if strict_cols:
            if in_cols_i != in_cols0 or tgt_cols_i != tgt_cols0:
                raise SystemExit(f"Model kolonları uyumsuz: {mpath}")
        else:
            # Esnek: kesişim üzerinden çalışmak istersen burada mapping kurabilirsin.
            pass

        # Her model kendi scaler'ı ile çalışır
        Xs_i = xsc_i.transform(X_norm)
        with torch.no_grad():
            Xs_t = torch.from_numpy(Xs_i.astype(np.float32)).to(device)
            Y_pred_s = model_i(Xs_t).cpu().numpy()
        Y_pred_i = ysc_i.inverse_transform(Y_pred_s)  # [N, D]
        preds.append(Y_pred_i)

    # Ortalama (uniform ensemble)
    Y_pred_ens = np.mean(np.stack(preds, axis=0), axis=0)

    # Y_true (rapor için)
    Y_true = df[tgt_cols0].to_numpy(dtype=float)

    # Raporu kur
    report_df = build_report_df(df, Y_true, Y_pred_ens, tgt_cols0, angles_use)

    # Kaydet
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    report_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"💾 Rapor kaydedildi: {out_csv}")

    # Konsol özeti
    summarize_to_console(report_df, angles_use)

    # JSON özet
    summary = {
        "rows": int(len(report_df)),
        "angles": angles_use,
        "de_means": {
            f"DE{a}": float(np.nanmean(report_df[f"DE{a}"])) if f"DE{a}" in report_df.columns else None
            for a in angles_use
        },
        "de_mean_overall": float(np.nanmean(report_df["DE_mean"])) if "DE_mean" in report_df.columns else None,
        "models": models
    }
    print("📝 Özet:", json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Batch predictor (ensemble destekli) + ΔE raporu")
    # Tek model
    ap.add_argument("--model", help="Tek model .pt yolu")
    # Ensemble
    ap.add_argument("--models", nargs="*", help="Birden fazla model path (ensemble)")
    ap.add_argument("--model_dir", help="Klasördeki *.pt modelleri (ensemble)")
    ap.add_argument("--models_manifest", help="JSON: {'models': ['.../fold1/best.pt', ...]}")
    # Veri
    ap.add_argument("--csv",   default="eval_datasetA.csv",     help="Değerlendirme CSV dosyası")
    ap.add_argument("--out",   default="report_ens.csv",        help="Çıktı rapor CSV")
    ap.add_argument("--angles", default="15,25,45,75,110",      help="Örn: 15,25,45,75,110 (boşsa varsayılan)")
    ap.add_argument("--strict_cols", action="store_true", help="Model kolonlarının birebir aynı olmasını zorunlu tut")
    args = ap.parse_args()

    paths = collect_model_paths(args.model, args.models, args.model_dir, args.models_manifest)
    if not paths:
        raise SystemExit("Model bulunamadı. --model / --models / --model_dir / --models_manifest verin.")

    run(models=paths, csv_path=args.csv, out_csv=args.out, angles_arg=args.angles, strict_cols=args.strict_cols)


if __name__ == "__main__":
    main()
