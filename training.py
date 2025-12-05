# -*- coding: utf-8 -*-
"""
Pigment -> Renk çok-çıktılı regresyon (SLIM)
===========================================
Bu sade sürüm; yalnızca **ham (row-normalized) pigment oranları** ile çalışır.
Kaldırılanlar:
- Audit / kNN outlier ve **örnek ağırlıkları** (performansı bozuyordu)
- Datapanel/ek özellikler (one-hot/top1/entropy vb.)
- CLR/ILR dönüşümleri (RAW ile daha iyi sonuç aldığınız için)
- Oversample, jitter augmentation, karmaşık DE ramp & açı ağırlıkları
- Aile etiketleme, temiz subset export vb.

Kalanlar (minimal ama üretim için yeterli):
- MLP ve (tercihen) MLP-ResNet mimarileri
- Huber/MSE ve ΔE76 karışımlı kayıp (sabit ağırlıkla)
- Erken durdurma (ΔE)
- CLI: train / predict / xai (SHAP)

Örnekler
--------
Eğitim:
  python pigment_torch_slim.py train --csv RS400.csv --save_dir ModelSave \
      --arch mlp_resnet --width 512 --depth 4 --block_hidden 1024 \
      --loss huber+de --de_weight 0.5 --de_angles 45 --patience 150 --plot_curve

Tek karışım tahmin:
  python pigment_torch_slim.py predict --model ModelSave/run/best.pt --pigments 821:50 910:50

Toplu tahmin:
  python pigment_torch_slim.py predict --model .../best.pt --input_csv mixes.csv --out_csv preds.csv

Global SHAP (ör. 45L için):
  python pigment_torch_slim.py xai --model .../best.pt --kind global --target 45L --csv RS400.csv
Local SHAP (tek reçete):
  python pigment_torch_slim.py xai --model .../best.pt --kind local --target 45L --pigments 821:50 910:50
"""
# -*- coding: utf-8 -*-


import os, json, argparse, datetime, csv, time, random, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# ---- Hedef ve giriş kolonları ----
TARGET_COLS = [
    "15L","25L","45L","75L","110L",
    "15a","25a","45a","75a","110a",
    "15b","25b","45b","75b","110b",
    "15Si","45Si","75Si",
    "15Sa","45Sa","75Sa",
    "G"
]

INPUT_COLS = [
    "726C","718Y","783S","772S","755C","744S","717C","980","910","135","828","835","831","826","836","838","815",
    "856","892","895","855","851","858","839","896","862","865","891","898","845","832","816","818","233","940",
    "424","356","384","306","318","331","198","154","290","522","322","570","410","550","530","371","482","343",
    "030","960","580","670","632","110","821","190"
]

ANGLES = ["15","25","45","75","110"]

# ---- Yardımcılar ----
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.fp = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fp, fieldnames=fieldnames)
        self.w.writeheader(); self.fp.flush()
    def log(self, **row):
        self.w.writerow(row); self.fp.flush()
    def close(self):
        try: self.fp.close()
        except: pass

# I/O temizliği
def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df

def ensure_columns(df, cols, fill=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def row_normalize(df, cols, eps=1e-12):
    sums = df[cols].sum(axis=1).replace(0, np.nan)
    for c in cols:
        df[c] = df[c] / (sums + eps)
    df[cols] = df[cols].fillna(0.0)
    return df

def to_tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)

# ΔE76
def cie76_delta_e(lab_true, lab_pred):
    diff = lab_true - lab_pred
    return np.sqrt((diff ** 2).sum(axis=1))


def predict_csv_ensemble(model_paths, input_csv, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(model_paths) > 0, "Model listesi boş."

    # İlk modelden kolon bilgisini alalım
    model0, xsc0, ysc0, in_cols0, out_cols0, meta0 = load_artifacts(model_paths[0], device)

    df = pd.read_csv(input_csv, sep=None, engine="python"); df.columns = [c.strip() for c in df.columns]
    df = ensure_columns(df, in_cols0, fill=0.0); df = coerce_numeric(df, in_cols0)

    # Satır-normalizasyon (eğitimle tutarlı)
    sums = df[in_cols0].sum(axis=1).replace(0, np.nan)
    for c in in_cols0:
        df[c] = df[c] / (sums + 1e-12)
    df[in_cols0] = df[in_cols0].fillna(0.0)

    X_raw = df[in_cols0].values.astype(np.float32)

    preds = []
    for mpath in model_paths:
        model, xsc, ysc, in_cols, out_cols, meta = load_artifacts(mpath, device)
        # Güvenlik: kolon tutarlılığı
        if in_cols != in_cols0 or out_cols != out_cols0:
            raise SystemExit(f"Model kolonları uyumsuz: {mpath}")
        Xs = xsc.transform(X_raw)
        with torch.no_grad():
            Yp_s = model(to_tensor(Xs, device)).cpu().numpy()
        Yp = ysc.inverse_transform(Yp_s)
        preds.append(Yp)

    Y_ens = np.mean(np.stack(preds, axis=0), axis=0)  # [N, D]
    final = pd.concat([df.reset_index(drop=True), pd.DataFrame(Y_ens, columns=out_cols0)], axis=1)
    final.to_csv(out_csv, index=False)
    print(f"💾 Ensemble tahminler: {out_csv} | modeller: {len(model_paths)}")
    return out_csv




# ---- Modeller ----

class PigmentTransformer(nn.Module):
    """
    Transformer-light: pigment oranlarını token gibi işler.
    - Her pigment için learnable embedding (E[i] ∈ R^{d_model})
    - Token = ratio_i * E[i] (+ opsiyonel ratio_proj)
    - Self-attention encoder katmanı (nhead, layers)
    - Readout: mean, weighted veya mean+weighted
    - Skip: ham girdiden çıkışa lineer atlama (baseline)
    """
    def __init__(self,
                 num_pigments: int,
                 out_dim: int,
                 d_model: int = 96,
                 nhead: int = 4,
                 num_layers: int = 2,
                 ff_mult: float = 2.0,
                 dropout: float = 0.08,
                 readout: str = "mean+weighted",
                 use_ratio_proj: bool = True,
                 mask_eps: float = 1e-12):
        super().__init__()
        self.num_pigments = num_pigments
        self.out_dim = out_dim
        self.d_model = d_model
        self.readout = readout
        self.mask_eps = mask_eps
        self.use_ratio_proj = use_ratio_proj

        # Her pigment için learnable embedding
        self.embed = nn.Parameter(torch.randn(num_pigments, d_model) * 0.02)

        # Oranı özellik olarak da enjekte etmek istersen:
        self.ratio_proj = nn.Linear(1, d_model) if use_ratio_proj else None

        # Transformer encoder
        d_ff = int(ff_mult * d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Readout boyutu
        read_dim = d_model
        if readout == "mean+weighted":
            read_dim = 2 * d_model
        elif readout in ("mean", "weighted"):
            read_dim = d_model
        else:
            raise ValueError(f"Unknown readout: {readout}")

        # Başlık
        self.head = nn.Sequential(
            nn.LayerNorm(read_dim),
            nn.Linear(read_dim, max(2 * out_dim, 128)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(2 * out_dim, 128), out_dim)
        )

        # Lineer skip (ham x -> out)
        self.skip = nn.Linear(num_pigments, out_dim, bias=False)

    def forward(self, x):  # x: [B, N], row-normalized pigment oranları
        B, N = x.shape
        assert N == self.num_pigments, "Pigment sayısı model ile uyumsuz."

        # [B, N, d_model]  tokenler
        tok = x.unsqueeze(-1) * self.embed.unsqueeze(0)  # ratio-gated embedding
        if self.ratio_proj is not None:
            tok = tok + self.ratio_proj(x.unsqueeze(-1))

        # Oranı 0 olan pigmentleri maskeye al (encoder onları görmezden gelir)
        # True => maskeli (yani yok say)
        key_padding_mask = (x <= self.mask_eps)  # [B, N], bool

        # Self-attention ile bağlamsallaştır
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # [B, N, d_model]

        # Readout
        if self.readout == "mean":
            g = h.mean(dim=1)  # [B, d_model]
        elif self.readout == "weighted":
            # Oranla ağırlıklı ortalama (oranlar zaten 1'e normalize edildi)
            g = (h * x.unsqueeze(-1)).sum(dim=1)
        else:  # mean+weighted
            g_mean = h.mean(dim=1)
            g_w = (h * x.unsqueeze(-1)).sum(dim=1)
            g = torch.cat([g_mean, g_w], dim=1)  # [B, 2*d_model]

        out = self.head(g) + self.skip(x)
        return out
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512,384,256), dropout=0.10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):
        return self.net(x) + self.skip(x)

class ResMLPBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1, norm='ln'):
        super().__init__(); Norm = nn.LayerNorm if norm=='ln' else nn.BatchNorm1d
        self.norm = Norm(dim); self.fc1 = nn.Linear(dim, hidden); self.act = nn.SiLU(); self.drop = nn.Dropout(dropout); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        h = self.norm(x); h = self.fc1(h); h = self.act(h); h = self.drop(h); h = self.fc2(h); return x + h

class MLPResNet(nn.Module):
    def __init__(self, in_dim, out_dim, width=512, depth=4, hidden=1024, dropout=0.1, norm='ln'):
        super().__init__()
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResMLPBlock(width, hidden=hidden, dropout=dropout, norm=norm) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width) if norm=='ln' else nn.BatchNorm1d(width), nn.Linear(width, out_dim))
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks: h = blk(h)
        return self.head(h) + self.skip(x)

# ---- Eğitim ----
def extract_triplet_cols(prefix):
    return f"{prefix}L", f"{prefix}a", f"{prefix}b"

def val_delta_e_stat(xy, model, y_scaler, target_cols, angles=("45",), stat="median"):
    x, y = xy; model.eval()
    with torch.no_grad(): yp_s = model(x).cpu().numpy()
    yt_s = y.cpu().numpy()
    yp = y_scaler.inverse_transform(yp_s); yt = y_scaler.inverse_transform(yt_s)
    vals = []
    for ang in angles:
        iL, ia, ib = target_cols.index(f"{ang}L"), target_cols.index(f"{ang}a"), target_cols.index(f"{ang}b")
        de = cie76_delta_e(yt[:, [iL, ia, ib]], yp[:, [iL, ia, ib]])
        vals.append(np.median(de) if stat=="median" else np.mean(de))
    return float(np.mean(vals))

def deltaE_loss_from_scaled(y_pred_s, y_true_s, y_mean, y_scale, triples,
                            comp_w=(1.0, 1.0, 1.0), angle_w=None, eps=1e-8):
    """
    comp_w = (wL, wa, wb)  -> ΔE'de L/a/b bileşen ağırlıkları
    angle_w: len(triples) uzunluğunda (örn. [w15,w25,w45,w75,w110])
    """
    y_pred = y_pred_s * y_scale + y_mean
    y_true = y_true_s * y_scale + y_mean
    wL, wa, wb = comp_w

    de_sum = 0.0
    denom = float(angle_w.sum().item() if torch.is_tensor(angle_w) else (sum(angle_w) if angle_w else len(triples)))

    for k, (iL, ia, ib) in enumerate(triples):
        dL = y_pred[:, iL] - y_true[:, iL]
        da = y_pred[:, ia] - y_true[:, ia]
        db = y_pred[:, ib] - y_true[:, ib]
        de = torch.sqrt(torch.clamp(wL*dL*dL + wa*da*da + wb*db*db, min=eps))  # [N]
        if angle_w is not None:
            w = angle_w[k] if torch.is_tensor(angle_w) else float(angle_w[k])
            de = de * w
        de_sum = de_sum + de
    return (de_sum / denom).mean()

def train_model(csv_path, save_dir, epochs=15000, batch_size=128, lr=1e-3,
                patience=150, seed=42, run_name="run", notes="",
                arch="mlp_resnet", hidden="512,384,256", width=512, depth=4, block_hidden=1024,
                arch_norm="ln", dropout=0.10,
                loss="huber+de", huber_beta=2.0, de_weight=0.5, de_angles="45",
                plot_curve=False, val_size=0.12, test_size=0.0001,
                ab_boost=1.0, nonlab_weight=1.0, de_angle_weights=None, lab_only=False,
                attn_d_model=96, attn_heads=4, attn_layers=2,
                attn_ff_mult=2.0, attn_readout="mean+weighted", attn_ratio_proj=True):


    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(save_dir, run_name.replace(" ", "_")); os.makedirs(exp_dir, exist_ok=True)

    # CSV
    df = pd.read_csv(csv_path, sep=None, engine="python"); df.columns = [c.strip() for c in df.columns]
    df = ensure_columns(df, INPUT_COLS + TARGET_COLS); df = coerce_numeric(df, INPUT_COLS + TARGET_COLS)

    # Hangi hedefler kullanılacak? (LAB-only ise Si/Sa/G'yi çıkar)
    target_cols = list(TARGET_COLS)
    if lab_only:
        target_cols = []
        for ang in ANGLES:
            for comp in ("L","a","b"):
                name = f"{ang}{comp}"
                if name in TARGET_COLS:
                    target_cols.append(name)

    # Sadece gerekli hedefler dolu olan satırlar
    df = df.dropna(subset=target_cols)
    # Girişleri eğitimle tutarlı normalize et
    df = row_normalize(df, INPUT_COLS)

    X = df[INPUT_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    X_trainval, X_test, y_trainval, y_test = train_split = train_test_split(X, y, test_size=test_size, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_rel, random_state=seed)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_s, X_val_s, X_test_s = x_scaler.transform(X_train), x_scaler.transform(X_val), x_scaler.transform(X_test)
    y_train_s, y_val_s, y_test_s = y_scaler.transform(y_train), y_scaler.transform(y_val), y_scaler.transform(y_test)

    # SHAP arka planı için küçük bir örneklem (opsiyonel)
    rng = np.random.default_rng(seed)
    bg_size = min(256, len(X_train_s))
    shap_bg_Xs = X_train_s[rng.choice(len(X_train_s), size=bg_size, replace=False)] if len(X_train_s) else None

    # Model
    if arch.lower() == "mlp_resnet":
        model = MLPResNet(
            in_dim=X.shape[1], out_dim=y.shape[1],
            width=width, depth=depth, hidden=block_hidden,
            dropout=dropout, norm=arch_norm
        ).to(device)
        arch_meta = dict(arch="mlp_resnet", width=width, depth=depth,
                            block_hidden=block_hidden, dropout=dropout, norm=arch_norm)

    elif arch.lower() == "attn":
        model = PigmentTransformer(
            num_pigments=X.shape[1], out_dim=y.shape[1],
            d_model=attn_d_model, nhead=attn_heads, num_layers=attn_layers,
            ff_mult=attn_ff_mult, dropout=dropout,
            readout=attn_readout, use_ratio_proj=attn_ratio_proj
        ).to(device)

        arch_meta = dict(
            arch="attn", d_model=attn_d_model, nhead=attn_heads,
            num_layers=attn_layers, ff_mult=attn_ff_mult,
            dropout=dropout, readout=attn_readout,
            use_ratio_proj=attn_ratio_proj
        )
    else:
        htuple = tuple(int(x) for x in hidden.split(",") if x.strip())
        model = MLP(in_dim=X.shape[1], out_dim=y.shape[1],
                    hidden=htuple, dropout=dropout).to(device)
        arch_meta = dict(arch="mlp", hidden=htuple, dropout=dropout)


    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # ---- ΔE için açı listesi ----
    angles_sel = [s.strip() for s in (de_angles.split(",") if isinstance(de_angles, str) and de_angles else ["45"])]

    # ---- ΔE üçlü indeksleri (L,a,b) -> target_cols üstünden ----
    triples = []
    for a in angles_sel:
        Ln, an, bn = f"{a}L", f"{a}a", f"{a}b"
        if (Ln in target_cols) and (an in target_cols) and (bn in target_cols):
            triples.append((target_cols.index(Ln), target_cols.index(an), target_cols.index(bn)))
    if len(triples) == 0:
        raise SystemExit("ΔE kaybı için L/a/b hedefleri bulunamadı. de_angles ve target_cols uyumunu kontrol et.")

    # ---- Çıkış-bazlı ağırlıklar (Huber/MSE tarafı) ----
    D = y_train_s.shape[1]
    w_vec = torch.ones(D, dtype=torch.float32, device=device)

    # a/b'yi güçlendir
    for ang in ANGLES:
        a_name, b_name = f"{ang}a", f"{ang}b"
        if a_name in target_cols: w_vec[target_cols.index(a_name)] *= float(ab_boost)
        if b_name in target_cols: w_vec[target_cols.index(b_name)] *= float(ab_boost)

    # Si/Sa/G'yi kıs (LAB-only isen zaten listede yok)
    for name in ["15Si","45Si","75Si","15Sa","45Sa","75Sa","G"]:
        if name in target_cols: w_vec[target_cols.index(name)] *= float(nonlab_weight)

    # ---- ΔE açı ağırlıkları ----
    angle_w = None
    if isinstance(de_angle_weights, str) and de_angle_weights.strip():
        amap = {}
        for kv in de_angle_weights.split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                amap[k.strip()] = float(v)
        angle_w = torch.tensor([amap.get(a, 1.0) for a in angles_sel], dtype=torch.float32, device=device)

    # ΔE bileşen ağırlıkları (L, a, b)
    comp_w = (1.0, float(ab_boost), float(ab_boost))

    # ---- Scaler tensörleri (ΔE için denormalize) ----
    y_mean_t  = torch.from_numpy(y_scaler.mean_.astype(np.float32)).to(device)
    y_scale_t = torch.from_numpy(y_scaler.scale_.astype(np.float32)).to(device)

    # ---- Kayıp fonksiyonları ----
    huber_none = nn.SmoothL1Loss(beta=huber_beta, reduction="none")

    def crit_mse(p, y):
        mse = (p - y) ** 2                 # [N,D]
        return (mse * w_vec).mean()

    def crit_huber(p, y):
        h = huber_none(p, y)               # [N,D]
        return (h * w_vec).mean()

    def crit_de(p, y):
        return deltaE_loss_from_scaled(
            p, y, y_mean_t, y_scale_t, triples,
            comp_w=comp_w, angle_w=angle_w
        )

    loss_name = (loss or "huber+de").lower()
    w_de = float(np.clip(de_weight, 0.0, 1.0))

    def compute_loss(pred, y):
        if loss_name == "mse": return crit_mse(pred, y)
        if loss_name == "huber": return crit_huber(pred, y)
        if loss_name in ("de","deltae"): return crit_de(pred, y)
        if loss_name in ("mse+de","mse_de"): return (1.0-w_de)*crit_mse(pred,y) + w_de*crit_de(pred,y)
        if loss_name in ("huber+de","huber_de"): return (1.0-w_de)*crit_huber(pred,y) + w_de*crit_de(pred,y)
        raise SystemExit(f"Bilinmeyen loss: {loss_name}")

    # Tensor paketleri
    train_tensor = (to_tensor(X_train_s, device), to_tensor(y_train_s, device))
    val_tensor   = (to_tensor(X_val_s, device),   to_tensor(y_val_s, device))
    test_tensor  = (to_tensor(X_test_s, device),  to_tensor(y_test_s, device))

    logger = CSVLogger(os.path.join(exp_dir, "metrics.csv"), ["epoch","train_loss","val_loss","val_de","best_de","patience_ctr","elapsed_sec"])

    best_val_de = float("inf"); best_state=None; patience_ctr=0; t0 = time.time()
    print(f"🖥️ Device: {device}")
    print(f"🚀 Eğitim | train={len(X_train_s)} val={len(X_val_s)} test={len(X_test_s)} | in={X.shape[1]} out={y.shape[1]} | params={param_count(model):,}")

    hist_train, hist_val = [], []
    for ep in range(1, epochs+1):
        # train
        model.train(); xtr,ytr = train_tensor
        perm = torch.randperm(xtr.shape[0], device=device)
        total=0.0; denom=0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]; xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(); pred = model(xb); loss_val = compute_loss(pred, yb); loss_val.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0); opt.step()
            total += float(loss_val.item()) * xb.size(0); denom += xb.size(0)
        tr = total / max(1,denom)

        # val
        model.eval(); xv,yv = val_tensor
        with torch.no_grad(): pv = model(xv); vl = float(compute_loss(pv, yv).item())
        # PATCH 5: target_cols + angles_sel
        vl_de = val_delta_e_stat(val_tensor, model, y_scaler, target_cols, angles=angles_sel, stat="median")
        hist_train.append(tr); hist_val.append(vl)

        # erken durdurma = ΔE
        improved = vl_de < best_val_de - 1e-9
        if improved:
            best_val_de = vl_de; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"model_state": best_state}, os.path.join(exp_dir, "best_weights.bin"))
            patience_ctr = 0
        else:
            patience_ctr += 1

        if ep % 20 == 0 or ep == 1:
            print(f"Ep {ep:04d} | Train {tr:.6f} | ValLoss {vl:.6f} | ValΔE[{','.join(angles_sel)}](median)={vl_de:.3f} | BestΔE {best_val_de:.3f} | patience {patience_ctr}/{patience}")
        logger.log(epoch=ep, train_loss=tr, val_loss=vl, val_de=vl_de, best_de=best_val_de, patience_ctr=patience_ctr, elapsed_sec=int(time.time()-t0))
        if patience_ctr >= patience: print("⏹️ Erken durdurma."); break

    logger.close()

    if best_state is not None: model.load_state_dict(best_state)

    # Test
    model.eval(); Xt,Yt = test_tensor
    with torch.no_grad(): Yp_s = model(Xt).cpu().numpy()
    Yt_s = Yt.cpu().numpy(); Yp = y_scaler.inverse_transform(Yp_s); Yt_den = y_scaler.inverse_transform(Yt_s)

    mse = float(np.mean((Yp - Yt_den)**2)); mae = float(np.mean(np.abs(Yp - Yt_den)))
    de_means = {}
    for ang in ANGLES:
        Lc, ac, bc = f"{ang}L", f"{ang}a", f"{ang}b"
        if (Lc in target_cols) and (ac in target_cols) and (bc in target_cols):
            iL, ia, ib = target_cols.index(Lc), target_cols.index(ac), target_cols.index(bc)
            de = cie76_delta_e(Yt_den[:, [iL, ia, ib]], Yp[:, [iL, ia, ib]])
            de_means[f"DE{ang}_mean"] = float(de.mean())
    print("✅ Test:", " | ".join([f"{k}={v:.3f}" for k,v in {**{"MSE":mse,"MAE":mae}, **de_means}.items()]))

    # Kaydet
    best_path = os.path.join(exp_dir, "best.pt")
    torch.save({
        "model_state": model.state_dict(),
        "input_cols": INPUT_COLS,
        "target_cols": target_cols,  # PATCH 5: kullanılan hedef listesi
        "in_dim": int(X.shape[1]),
        "x_scaler_mean": x_scaler.mean_.astype(np.float32),
        "x_scaler_scale": x_scaler.scale_.astype(np.float32),
        "y_scaler_mean": y_scaler.mean_.astype(np.float32),
        "y_scaler_scale": y_scaler.scale_.astype(np.float32),
        "shap_bg_Xs": shap_bg_Xs.astype(np.float32) if shap_bg_Xs is not None else None,
        **arch_meta,
        "run_name": run_name, "notes": notes
    }, best_path)

    summary = {"best_val_de": best_val_de, "test_mse": mse, "test_mae": mae, **de_means, "epochs_ran": len(hist_train), "finished_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)

    if plot_curve:
        try:
            import matplotlib.pyplot as plt
            plt.figure(); plt.plot(hist_train, label="train"); plt.plot(hist_val, label="val"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout();
            plt.savefig(os.path.join(exp_dir, "loss_curve.png"), dpi=150); plt.close()
        except Exception as e:
            print("Plot çizilemedi:", e)

    print(f"💾 Kayıt klasörü: {exp_dir}\n💾 En iyi model: {best_path}")
    return best_path



def train_kfold(csv_path, save_dir, cv_folds=5, cv_seed=42,
                epochs=15000, batch_size=128, lr=1e-3, patience=150, seed=42,
                run_name="run", notes="", plot_curve=False,
                arch="mlp_resnet", hidden="512,384,256", width=512, depth=4, block_hidden=1024, arch_norm="ln", dropout=0.10,
                loss="huber+de", huber_beta=2.0, de_weight=0.5, de_angles="45",
                ab_boost=1.0, nonlab_weight=1.0, de_angle_weights=None, lab_only=False,
                attn_d_model=96, attn_heads=4, attn_layers=2, attn_ff_mult=2.0, attn_readout="mean+weighted", attn_ratio_proj=True):

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CSV hazirlik (train_model ile birebir) ---
    df = pd.read_csv(csv_path, sep=None, engine="python"); df.columns = [c.strip() for c in df.columns]
    df = ensure_columns(df, INPUT_COLS + TARGET_COLS); df = coerce_numeric(df, INPUT_COLS + TARGET_COLS)

    # LAB-only hedef listesi
    target_cols = list(TARGET_COLS)
    if lab_only:
        target_cols = []
        for ang in ANGLES:
            for comp in ("L","a","b"):
                name = f"{ang}{comp}"
                if name in TARGET_COLS: target_cols.append(name)

    df = df.dropna(subset=target_cols)
    df = row_normalize(df, INPUT_COLS)

    X = df[INPUT_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    kf = KFold(n_splits=int(cv_folds), shuffle=True, random_state=cv_seed)
    manifest = {"models": [], "run_name": run_name, "cv_folds": int(cv_folds)}

    fold_no = 0
    for tr_idx, va_idx in kf.split(X):
        fold_no += 1
        exp_dir = os.path.join(save_dir, f"{run_name}_fold{fold_no}")
        os.makedirs(exp_dir, exist_ok=True)

        X_tr, Y_tr = X[tr_idx], y[tr_idx]
        X_va, Y_va = X[va_idx], y[va_idx]

        x_scaler = StandardScaler().fit(X_tr)
        y_scaler = StandardScaler().fit(Y_tr)
        X_tr_s, X_va_s = x_scaler.transform(X_tr), x_scaler.transform(X_va)
        Y_tr_s, Y_va_s = y_scaler.transform(Y_tr), y_scaler.transform(Y_va)

        # SHAP bg örneklemi
        rng = np.random.default_rng(seed)
        bg_size = min(256, len(X_tr_s))
        shap_bg_Xs = X_tr_s[rng.choice(len(X_tr_s), size=bg_size, replace=False)] if len(X_tr_s) else None

        # --- Model kurulum (mevcut mantik) ---
        if arch.lower() == "mlp_resnet":
            model = MLPResNet(in_dim=X.shape[1], out_dim=y.shape[1],
                              width=width, depth=depth, hidden=block_hidden,
                              dropout=dropout, norm=arch_norm).to(device)
            arch_meta = dict(arch="mlp_resnet", width=width, depth=depth,
                             block_hidden=block_hidden, dropout=dropout, norm=arch_norm)
        elif arch.lower() == "attn":
            model = PigmentTransformer(num_pigments=X.shape[1], out_dim=y.shape[1],
                                       d_model=attn_d_model, nhead=attn_heads, num_layers=attn_layers,
                                       ff_mult=attn_ff_mult, dropout=dropout,
                                       readout=attn_readout, use_ratio_proj=attn_ratio_proj).to(device)
            arch_meta = dict(arch="attn", d_model=attn_d_model, nhead=attn_heads,
                             num_layers=attn_layers, ff_mult=attn_ff_mult,
                             dropout=dropout, readout=attn_readout, use_ratio_proj=attn_ratio_proj)
        else:
            htuple = tuple(int(x) for x in hidden.split(",") if x.strip())
            model = MLP(in_dim=X.shape[1], out_dim=y.shape[1],
                        hidden=htuple, dropout=dropout).to(device)
            arch_meta = dict(arch="mlp", hidden=htuple, dropout=dropout)

        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # --- ΔE açılar & üçlü indeksler ---
        angles_sel = [s.strip() for s in (de_angles.split(",") if isinstance(de_angles, str) and de_angles else ["45"])]
        triples = []
        for a in angles_sel:
            Ln, an, bn = f"{a}L", f"{a}a", f"{a}b"
            if (Ln in target_cols) and (an in target_cols) and (bn in target_cols):
                triples.append((target_cols.index(Ln), target_cols.index(an), target_cols.index(bn)))
        if len(triples) == 0:
            raise SystemExit("ΔE kaybı için L/a/b hedefleri bulunamadı. de_angles ve target_cols uyumunu kontrol et.")

        # --- Huber/MSE çıkış ağırlıkları ---
        D = Y_tr_s.shape[1]
        w_vec = torch.ones(D, dtype=torch.float32, device=device)
        for ang in ANGLES:
            a_name, b_name = f"{ang}a", f"{ang}b"
            if a_name in target_cols: w_vec[target_cols.index(a_name)] *= float(ab_boost)
            if b_name in target_cols: w_vec[target_cols.index(b_name)] *= float(ab_boost)
        for name in ["15Si","45Si","75Si","15Sa","45Sa","75Sa","G"]:
            if name in target_cols: w_vec[target_cols.index(name)] *= float(nonlab_weight)

        # --- Açı ağırlıkları ---
        angle_w = None
        if isinstance(de_angle_weights, str) and de_angle_weights.strip():
            amap = {}
            for kv in de_angle_weights.split(","):
                if ":" in kv:
                    k, v = kv.split(":", 1); amap[k.strip()] = float(v)
            angle_w = torch.tensor([amap.get(a, 1.0) for a in angles_sel], dtype=torch.float32, device=device)

        comp_w = (1.0, float(ab_boost), float(ab_boost))
        y_mean_t  = torch.from_numpy(y_scaler.mean_.astype(np.float32)).to(device)
        y_scale_t = torch.from_numpy(y_scaler.scale_.astype(np.float32)).to(device)

        huber_none = nn.SmoothL1Loss(beta=huber_beta, reduction="none")
        def crit_mse(p, y):
            mse = (p - y) ** 2
            return (mse * w_vec).mean()
        def crit_huber(p, y):
            h = huber_none(p, y)
            return (h * w_vec).mean()
        def crit_de(p, y):
            return deltaE_loss_from_scaled(p, y, y_mean_t, y_scale_t, triples, comp_w=comp_w, angle_w=angle_w)

        loss_name = (loss or "huber+de").lower()
        w_de = float(np.clip(de_weight, 0.0, 1.0))
        def compute_loss(pred, ytrue):
            if loss_name == "mse": return crit_mse(pred, ytrue)
            if loss_name == "huber": return crit_huber(pred, ytrue)
            if loss_name in ("de","deltae"): return crit_de(pred, ytrue)
            if loss_name in ("mse+de","mse_de"): return (1.0-w_de)*crit_mse(pred,ytrue) + w_de*crit_de(pred,ytrue)
            if loss_name in ("huber+de","huber_de"): return (1.0-w_de)*crit_huber(pred,ytrue) + w_de*crit_de(pred,ytrue)
            raise SystemExit(f"Bilinmeyen loss: {loss_name}")

        # Tensor paketleri
        train_tensor = (to_tensor(X_tr_s, device), to_tensor(Y_tr_s, device))
        val_tensor   = (to_tensor(X_va_s, device), to_tensor(Y_va_s, device))

        logger = CSVLogger(os.path.join(exp_dir, "metrics.csv"),
                           ["epoch","train_loss","val_loss","val_de","best_de","patience_ctr","elapsed_sec"])
        best_val_de = float("inf"); best_state=None; patience_ctr=0; t0 = time.time()
        print(f"🖥️ Device: {device} | Fold {fold_no}/{cv_folds}")
        print(f"🚀 Eğitim | train={len(X_tr_s)} val={len(X_va_s)} | in={X.shape[1]} out={y.shape[1]} | params={param_count(model):,}")

        for ep in range(1, epochs+1):
            model.train(); xtr,ytr = train_tensor
            perm = torch.randperm(xtr.shape[0], device=device)
            total=0.0; denom=0
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i+batch_size]; xb, yb = xtr[idx], ytr[idx]
                opt.zero_grad(); pred = model(xb); loss_val = compute_loss(pred, yb); loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0); opt.step()
                total += float(loss_val.item()) * xb.size(0); denom += xb.size(0)
            tr = total / max(1,denom)

            model.eval(); xv,yv = val_tensor
            with torch.no_grad(): pv = model(xv); vl = float(compute_loss(pv, yv).item())
            vl_de = val_delta_e_stat(val_tensor, model, y_scaler, target_cols, angles=angles_sel, stat="median")

            improved = vl_de < best_val_de - 1e-9
            if improved:
                best_val_de = vl_de
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({"model_state": best_state}, os.path.join(exp_dir, "best_weights.bin"))
                patience_ctr = 0
            else:
                patience_ctr += 1

            if ep % 20 == 0 or ep == 1:
                print(f"[Fold {fold_no}] Ep {ep:04d} | Train {tr:.6f} | ValLoss {vl:.6f} | ValΔE[{','.join(angles_sel)}](med)={vl_de:.3f} | BestΔE {best_val_de:.3f} | patience {patience_ctr}/{patience}")
            logger.log(epoch=ep, train_loss=tr, val_loss=vl, val_de=vl_de, best_de=best_val_de, patience_ctr=patience_ctr, elapsed_sec=int(time.time()-t0))
            if patience_ctr >= patience:
                print(f"[Fold {fold_no}] ⏹️ Erken durdurma."); break
        logger.close()

        if best_state is not None: model.load_state_dict(best_state)

        # Kaydet
        best_path = os.path.join(exp_dir, "best.pt")
        torch.save({
            "model_state": model.state_dict(),
            "input_cols": INPUT_COLS,
            "target_cols": target_cols,
            "in_dim": int(X.shape[1]),
            "x_scaler_mean": x_scaler.mean_.astype(np.float32),
            "x_scaler_scale": x_scaler.scale_.astype(np.float32),
            "y_scaler_mean": y_scaler.mean_.astype(np.float32),
            "y_scaler_scale": y_scaler.scale_.astype(np.float32),
            "shap_bg_Xs": shap_bg_Xs.astype(np.float32) if shap_bg_Xs is not None else None,
            **arch_meta,
            "run_name": f"{run_name}_fold{fold_no}", "notes": notes
        }, best_path)

        manifest["models"].append(best_path)
        print(f"[Fold {fold_no}] 💾 En iyi model: {best_path}")

    manifest_path = os.path.join(save_dir, f"{run_name}_ensemble.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"🧩 Ensemble manifest kaydedildi: {manifest_path}")

    return manifest["models"]





# ---- Yükle & Tahmin ----
def load_artifacts(model_path, device):
    blob = torch.load(model_path, map_location=device, weights_only=False)
    input_cols  = list(blob["input_cols"])  # list'e zorla
    target_cols = list(blob["target_cols"])
    state       = blob["model_state"]
    in_dim      = int(blob.get("in_dim", len(input_cols)))
    arch = (blob.get("arch") or blob.get("arch_meta", {}).get("arch") or blob.get("run_arch") or "mlp_resnet").lower()

    if arch == "mlp_resnet":
        width        = int(blob.get("width", 512))
        depth        = int(blob.get("depth", 4))
        block_hidden = int(blob.get("block_hidden", 1024))
        dropout      = float(blob.get("dropout", 0.10))
        norm         = blob.get("norm", "ln")
        model = MLPResNet(in_dim=in_dim, out_dim=len(target_cols), width=width, depth=depth, hidden=block_hidden, dropout=dropout, norm=norm).to(device)
    elif arch == "attn":
        d_model       = int(blob.get("d_model", 96))
        nhead         = int(blob.get("nhead", 4))
        num_layers    = int(blob.get("num_layers", 2))
        ff_mult       = float(blob.get("ff_mult", 2.0))
        dropout       = float(blob.get("dropout", 0.08))
        readout       = blob.get("readout", "mean+weighted")
        use_ratio_proj= bool(blob.get("use_ratio_proj", True))

        model = PigmentTransformer(
            num_pigments=in_dim, out_dim=len(target_cols),
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_mult=ff_mult, dropout=dropout,
            readout=readout, use_ratio_proj=use_ratio_proj
        ).to(device)
    else:
        hidden  = tuple(blob.get("hidden", (512,384,256)))
        dropout = float(blob.get("dropout", 0.10))
        model = MLP(in_dim=in_dim, out_dim=len(target_cols), hidden=hidden, dropout=dropout).to(device)

    model.load_state_dict(state, strict=True); model.eval()

    x_scaler = StandardScaler(); y_scaler = StandardScaler()
    x_scaler.mean_  = np.array(blob["x_scaler_mean"], dtype=np.float64)
    x_scaler.scale_ = np.array(blob["x_scaler_scale"], dtype=np.float64)
    y_scaler.mean_  = np.array(blob["y_scaler_mean"], dtype=np.float64)
    y_scaler.scale_ = np.array(blob["y_scaler_scale"], dtype=np.float64)
    x_scaler.var_ = x_scaler.scale_ ** 2; y_scaler.var_ = y_scaler.scale_ ** 2

    return model, x_scaler, y_scaler, input_cols, target_cols, blob

def parse_pigments(pairs):
    out = {}; pairs = pairs or []
    for p in pairs:
        if ":" not in p: continue
        k, v = p.split(":", 1); k = k.strip()
        try: out[k] = float(str(v).replace(",", "."))  # 10,5 -> 10.5
        except: out[k] = 0.0
    return out

def vectorize_mixture(mix_dict, input_cols):
    v = np.zeros(len(input_cols), dtype=np.float32)
    for i, col in enumerate(input_cols): v[i] = float(mix_dict.get(col, 0.0))
    s = v.sum(); v = (v / s) if s > 0 else v
    return v.reshape(1, -1)

def predict_single(model_path, pigment_pairs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, xsc, ysc, in_cols, out_cols, meta = load_artifacts(model_path, device)
    x_raw = vectorize_mixture(parse_pigments(pigment_pairs), in_cols)  # zaten row-normalize
    xs = xsc.transform(x_raw)
    with torch.no_grad(): y_pred_s = model(to_tensor(xs, device)).cpu().numpy()
    y_pred = ysc.inverse_transform(y_pred_s)[0]
    res = {col: float(val) for col, val in zip(out_cols, y_pred)}
    print("📌 Özet:", json.dumps({k: res[k] for k in ["15L","25L","45L","75L","110L","G"] if k in res}, ensure_ascii=False, indent=2))
    return res

def predict_csv(model_path, input_csv, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, xsc, ysc, in_cols, out_cols, meta = load_artifacts(model_path, device)
    df = pd.read_csv(input_csv, sep=None, engine="python"); df.columns = [c.strip() for c in df.columns]
    df = ensure_columns(df, in_cols, fill=0.0); df = coerce_numeric(df, in_cols)

    # PATCH 6: Eğitimle tutarlı şekilde satır-normalizasyon
    sums = df[in_cols].sum(axis=1).replace(0, np.nan)
    for c in in_cols:
        df[c] = df[c] / (sums + 1e-12)
    df[in_cols] = df[in_cols].fillna(0.0)

    X_raw  = df[in_cols].values.astype(np.float32)
    Xs     = xsc.transform(X_raw)
    with torch.no_grad(): Yp = model(to_tensor(Xs, device)).cpu().numpy()
    Yp_den = ysc.inverse_transform(Yp)
    final = pd.concat([df.reset_index(drop=True), pd.DataFrame(Yp_den, columns=out_cols)], axis=1)
    final.to_csv(out_csv, index=False); print(f"💾 Tahminler: {out_csv}")
    return out_csv

# ---- XAI (SHAP) ----
def _target_index(target_cols, name: str) -> int:
    try: return target_cols.index(name)
    except ValueError: raise SystemExit(f"Hedef bulunamadı: {name}")

def _sv_to_2d(sv, out_idx=0):
    import numpy as _np
    try: vals = sv.values
    except AttributeError: vals = _np.asarray(sv)
    vals = _np.asarray(vals)
    if vals.ndim == 3: vals = vals[:, out_idx, :]
    elif vals.ndim == 2: pass
    elif vals.ndim == 1: vals = vals.reshape(1, -1)
    else: raise ValueError(f"Beklenmeyen SHAP shape: {vals.shape}")
    return vals

def xai_global(model_path, csv_path=None, target="45L", samples=256, out_dir="xai_outputs"):
    import shap, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, xsc, ysc, in_cols, out_cols, meta = load_artifacts(model_path, device)
    out_idx = _target_index(out_cols, target)

    bg = meta.get("shap_bg_Xs", None)
    if bg is None or (isinstance(bg, np.ndarray) and bg.size == 0):
        if not csv_path: raise SystemExit("SHAP background yok. --csv veriniz.")
        df = pd.read_csv(csv_path, sep=None, engine="python"); df = ensure_columns(df, in_cols, fill=0.0); df = coerce_numeric(df, in_cols)
        X = df[in_cols].values.astype(np.float32); bg = xsc.transform(X)
    if len(bg) > samples: bg = bg[:samples]

    y_mean = float(ysc.mean_[out_idx]); y_std = float(ysc.scale_[out_idx])
    def f_model(Xs_np):
        Xs = torch.from_numpy(Xs_np.astype(np.float32)).to(device)
        with torch.no_grad(): Y_s = model(Xs).detach().cpu().numpy()
        return (Y_s[:, [out_idx]] * y_std + y_mean)

    explainer = shap.Explainer(f_model, bg)
    sv = explainer(bg)
    vals_2d = _sv_to_2d(sv, out_idx=0)
    feat_names = in_cols

    shap.summary_plot(vals_2d, features=bg, feature_names=feat_names, show=False, plot_type="bar"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"summary_bar_{target}.png"), dpi=150); plt.close()
    shap.summary_plot(vals_2d, features=bg, feature_names=feat_names, show=False); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"summary_beeswarm_{target}.png"), dpi=150); plt.close()
    print(f"📝 Global SHAP: {out_dir}")

def xai_local(model_path, pigment_pairs, target="45L", topk=12, out_dir="xai_outputs", csv_path=None, samples=512):
    import shap, matplotlib.pyplot as plt, numpy as np
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, xsc, ysc, in_cols, out_cols, meta = load_artifacts(model_path, device)
    out_idx = _target_index(out_cols, target)

    bg = meta.get("shap_bg_Xs", None)
    if bg is None or (isinstance(bg, np.ndarray) and bg.size == 0):
        if not csv_path: raise SystemExit("SHAP background yok. --csv veriniz.")
        df = pd.read_csv(csv_path, sep=None, engine="python"); df = ensure_columns(df, in_cols, fill=0.0); df = coerce_numeric(df, in_cols)
        X = df[in_cols].values.astype(np.float32); bg = xsc.transform(X)
        if len(bg) > samples: bg = bg[:samples]

    mix = parse_pigments(pigment_pairs); x_row = xsc.transform(vectorize_mixture(mix, in_cols)).astype(np.float32)
    y_mean = float(ysc.mean_[out_idx]); y_std = float(ysc.scale_[out_idx])
    def f_model(Xs_np):
        Xs = torch.from_numpy(Xs_np.astype(np.float32)).to(device)
        with torch.no_grad(): Y_s = model(Xs).detach().cpu().numpy()
        return (Y_s[:, [out_idx]] * y_std + y_mean)

    masker = shap.maskers.Independent(bg); explainer = shap.Explainer(f_model, masker, algorithm="permutation")
    sv = explainer(x_row)

    def _vec(sv_obj):
        try: vals = sv_obj.values
        except AttributeError: vals = np.asarray(sv_obj)
        vals = np.asarray(vals)
        if vals.ndim == 3: vals = vals[0,0,:]
        elif vals.ndim == 2: vals = vals[0]
        elif vals.ndim == 1: pass
        else: raise SystemExit("SHAP shape beklenmedik")
        return vals

    vals = _vec(sv); basev = np.asarray(getattr(sv, "base_values", 0)).reshape(-1)[0]; feats = x_row[0]
    kk = max(1, min(int(topk), len(vals))); idx = np.argsort(np.abs(vals))[::-1][:kk]

    shap.plots.waterfall(shap.Explanation(values=vals[idx], base_values=basev, data=feats[idx], feature_names=[in_cols[i] for i in idx]), show=False)
    plt.tight_layout(); out_png = os.path.join(out_dir, f"local_waterfall_{target}.png"); plt.savefig(out_png, dpi=150); plt.close()
    print(f"🔍 Local SHAP: {out_png}")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Pigment -> Renk (SLIM)")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_tr = sub.add_parser("train", help="Model eğit")
    ap_tr.add_argument("--csv", required=True)
    ap_tr.add_argument("--save_dir", default="models")
    ap_tr.add_argument("--epochs", type=int, default=15000)
    ap_tr.add_argument("--batch_size", type=int, default=256)
    ap_tr.add_argument("--lr", type=float, default=1e-3)
    ap_tr.add_argument("--patience", type=int, default=200)
    ap_tr.add_argument("--seed", type=int, default=42)
    ap_tr.add_argument("--run_name", default="run")
    ap_tr.add_argument("--notes", default="")
    ap_tr.add_argument("--plot_curve", action="store_true")

    ap_tr.add_argument("--arch", choices=["mlp","mlp_resnet","attn"], default="mlp_resnet")
    ap_tr.add_argument("--hidden", default="512,384,256")
    ap_tr.add_argument("--width", type=int, default=312)
    ap_tr.add_argument("--depth", type=int, default=3)
    ap_tr.add_argument("--block_hidden", type=int, default=712)
    ap_tr.add_argument("--arch_norm", choices=["ln","bn"], default="bn")
    ap_tr.add_argument("--dropout", type=float, default=0.15)

    ap_tr.add_argument("--loss", choices=["mse","huber","de","mse+de","huber+de"], default="huber+de")
    ap_tr.add_argument("--huber_beta", type=float, default=2.0)
    ap_tr.add_argument("--de_weight", type=float, default=1.0)
    ap_tr.add_argument("--de_angles", default="15,25,45,75,110")

    ap_tr.add_argument("--ab_boost", type=float, default=2.4, help="Huber/MSE tarafında a/b hatasını çarpan")
    ap_tr.add_argument("--nonlab_weight", type=float, default=1.0, help="Si/Sa/G için ağırlık (1.0=standart, <1.0 kısar)")
    ap_tr.add_argument("--de_angle_weights", default="15:1.0,25:1.0,45:1.0,75:1.0,110:1.0", help='Açı ağırlıkları: "15:1.7,25:1.3,45:1.0,75:1.2,110:1.2"')
    ap_tr.add_argument("--lab_only", action="store_true", help="Sadece LAB hedefleriyle eğitim (Si/Sa/G çıkarılır)")

    ap_tr.add_argument("--attn_d_model", type=int, default=96)
    ap_tr.add_argument("--attn_heads", type=int, default=8)
    ap_tr.add_argument("--attn_layers", type=int, default=3)
    ap_tr.add_argument("--attn_ff_mult", type=float, default=2.0)
    ap_tr.add_argument("--attn_readout", choices=["mean","weighted","mean+weighted"], default="mean+weighted")
    ap_tr.add_argument("--attn_ratio_proj", action="store_true", help="Tokenlara oran-proj ekle (kapamak için bayrağı kullanma)")

    ap_tr.add_argument("--cv_folds", type=int, default=5, help=">1 ise K-fold cross-val ile eğitim")
    ap_tr.add_argument("--cv_seed", type=int, default=42)


    ap_pr = sub.add_parser("predict", help="Tahmin al")
    ap_pr.add_argument("--model", required=True)
    ap_pr.add_argument("--pigments", nargs="*")
    ap_pr.add_argument("--input_csv")
    ap_pr.add_argument("--out_csv", default="predictions.csv")
    ap_pr.add_argument("--models", nargs="*", help="Birden fazla model path (ensemble)")
    ap_pr.add_argument("--model_dir", help="Klasördeki *.pt modelleri (ensemble)")
    ap_pr.add_argument("--models_manifest", help="JSON: {'models': ['.../fold1/best.pt', ...]}")


    ap_x = sub.add_parser("xai", help="SHAP açıklamaları")
    ap_x.add_argument("--model", required=True)
    ap_x.add_argument("--kind", choices=["global","local"], default="global")
    ap_x.add_argument("--target", default="45L")
    ap_x.add_argument("--csv")
    ap_x.add_argument("--samples", type=int, default=256)
    ap_x.add_argument("--pigments", nargs="*")
    ap_x.add_argument("--topk", type=int, default=12)
    ap_x.add_argument("--out_dir", default="xai_outputs")

    args = ap.parse_args()

    if args.mode == "train":
        if args.cv_folds and args.cv_folds > 1:
            train_kfold(
                csv_path=args.csv, save_dir=args.save_dir, cv_folds=args.cv_folds, cv_seed=args.cv_seed,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, seed=args.seed,
                run_name=args.run_name, notes=args.notes, plot_curve=args.plot_curve,
                arch=args.arch, hidden=args.hidden, width=args.width, depth=args.depth, block_hidden=args.block_hidden, arch_norm=args.arch_norm, dropout=args.dropout,
                loss=args.loss, huber_beta=args.huber_beta, de_weight=args.de_weight, de_angles=args.de_angles,
                ab_boost=args.ab_boost, nonlab_weight=args.nonlab_weight, de_angle_weights=args.de_angle_weights, lab_only=args.lab_only,
                attn_d_model=args.attn_d_model, attn_heads=args.attn_heads, attn_layers=args.attn_layers, attn_ff_mult=args.attn_ff_mult,
                attn_readout=args.attn_readout, attn_ratio_proj=args.attn_ratio_proj
            )
        else:
            train_model(
                csv_path=args.csv, save_dir=args.save_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                patience=args.patience, seed=args.seed, run_name=args.run_name, notes=args.notes, plot_curve=args.plot_curve,
                arch=args.arch, hidden=args.hidden, width=args.width, depth=args.depth, block_hidden=args.block_hidden, arch_norm=args.arch_norm, dropout=args.dropout,
                loss=args.loss, huber_beta=args.huber_beta, de_weight=args.de_weight, de_angles=args.de_angles,
                ab_boost=args.ab_boost, nonlab_weight=args.nonlab_weight, de_angle_weights=args.de_angle_weights, lab_only=args.lab_only,
                attn_d_model=args.attn_d_model, attn_heads=args.attn_heads, attn_layers=args.attn_layers, attn_ff_mult=args.attn_ff_mult,
                attn_readout=args.attn_readout, attn_ratio_proj=args.attn_ratio_proj
            )
            
    elif args.mode == "predict":
        # Ensemble?
        if (args.models and len(args.models) > 0) or args.model_dir or args.models_manifest:
            import glob
            paths = []
            if args.models: paths.extend(args.models)
            if args.model_dir: paths.extend(sorted(glob.glob(os.path.join(args.model_dir, "*.pt"))))
            if args.models_manifest:
                with open(args.models_manifest, "r", encoding="utf-8") as f:
                    js = json.load(f)
                    paths.extend(js.get("models", []))
            paths = [p for p in paths if p and os.path.exists(p)]
            if not paths:
                raise SystemExit("Ensemble için model bulunamadı.")
            if args.input_csv:
                predict_csv_ensemble(paths, args.input_csv, args.out_csv)
            else:
                raise SystemExit("Ensemble modda --input_csv gereklidir.")
        else:
            # Tek model davranışı (mevcut)
            if args.input_csv:
                predict_csv(args.model, args.input_csv, args.out_csv)
            elif args.pigments:
                res = predict_single(args.model, args.pigments); print(json.dumps(res, ensure_ascii=False, indent=2))
            else:
                raise SystemExit("Pigments veya input_csv belirtin.")

    elif args.mode == "xai":
        if args.kind == "global":
            xai_global(args.model, csv_path=args.csv, target=args.target, samples=args.samples, out_dir=args.out_dir)
        else:
            if not args.pigments: raise SystemExit("Local XAI için --pigments gerekli.")
            xai_local(args.model, pigment_pairs=args.pigments, target=args.target, topk=args.topk, out_dir=args.out_dir, csv_path=args.csv, samples=args.samples)

if __name__ == "__main__":
    main()
