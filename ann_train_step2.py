#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 2  -  ANN (full input) training
================================================================================
Trains two surrogate-model architectures on the exergy campaign dataset:

  (1) a SINGLE multi-output network  : 26 inputs -> 18 outputs
  (2) SIX per-section networks       : 26 inputs -> 3 outputs each
                                       (I_B, eps, IP for one section)

Both are feed-forward MLPs (Keras): two hidden layers (64, 32) with tanh,
linear output, Adam(1e-3), MSE loss, batch 32, early stopping (patience 30).
Inputs and outputs are standardized (StandardScaler, fit on TRAIN only).
Data: rows with status=="OK" (the 411 clean samples). 70/15/15 split, seed 42.

Outputs (saved into  <ASPEN_FOLDER>/ANN_models/ ):
  single_18output.keras
  section_<Name>.keras            (x6)
  scaler_X.pkl, scaler_y.pkl       (single-net output scaler)
  scaler_y_<Name>.pkl              (per-section output scalers)
  metrics_report.csv               (R2/MSE/RMSE per output, both architectures)
  training_summary.txt

RUN:  pip install tensorflow scikit-learn pandas joblib
      python ann_train_step2.py
================================================================================
"""
import os, sys, json, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models")

SEED         = 42
TEST_FRAC    = 0.15
VAL_FRAC     = 0.15          # of the full set; train gets the remaining 0.70
HIDDEN       = (64, 32)
ACTIVATION   = "tanh"
LR           = 1e-3
BATCH        = 32
MAX_EPOCHS   = 500
PATIENCE     = 30
USE_STATUS   = ("OK",)       # train only on clean rows (the 411). Add "OK_CAPPED"
                             # here if you later choose to include capped samples.

# 26 input feature columns (exact CSV headers)
INPUT_COLS = [
 "rawmeal_feed[kg/s]","rmgas_flow[kg/s]","rmgas_temp[K]","coal_feed[kg/s]",
 "cmgas_flow[kg/s]","cmgas_temp[K]","clrcair_flow[kg/s]","clrcair2_flow[kg/s]",
 "rmfan_T[K]","rmdry_T[K]","gasht_T[K]","cmdry_T[K]",
 "phk1_T[K]","phk2_T[K]","phk3_T[K]","phk4_T[K]","phk5_T[K]",
 "phc1_T[K]","phc2_T[K]","phc3_T[K]","phc4_T[K]","phc5_T[K]",
 "clcalc_T[K]","coal_split[-]","secair_split[-]","meal_split[-]",
]

# 6 sections -> their 3 output columns each (18 total)
SECTIONS = ["RawMill","CoalMill","Preheater","Calciner","Kiln","ClinkerCooler"]
def sec_outputs(s): return ["%s_I_B_MW"%s, "%s_eps"%s, "%s_IP_MW"%s]
OUTPUT_COLS = [c for s in SECTIONS for c in sec_outputs(s)]   # 18, section-blocked
# 19th output: clinker product exergy (MW) - enables specific-irreversibility
# optimisation in Step 4 (total irreversibility per unit clinker exergy).
EXTRA_OUTPUTS = ["clinker_ex_MW"]
OUTPUT_COLS = OUTPUT_COLS + EXTRA_OUTPUTS                      # 19 total

# ============================================================ SETUP ============
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ----- publication plotting (Times New Roman, size 12) ----------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
def _setup_fonts():
    # Prefer Times New Roman (present on Windows); fall back gracefully.
    for fam in ["Times New Roman","Times","DejaVu Serif"]:
        if any(f.name==fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]=fam; break
    else:
        plt.rcParams["font.family"]="serif"
    plt.rcParams.update({
        "font.size":12, "axes.titlesize":12, "axes.labelsize":12,
        "xtick.labelsize":12, "ytick.labelsize":12, "legend.fontsize":12,
        "figure.dpi":120, "savefig.dpi":300, "axes.linewidth":0.8,
    })
_setup_fonts()
PLOT_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_parity(y_true, y_pred, names, arch_tag):
    """One predicted-vs-actual parity panel per output, with R2 annotated."""
    n=len(names); ncol=3; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*3.2,nrow*3.0))
    axes=np.array(axes).reshape(-1)
    for j,nm in enumerate(names):
        ax=axes[j]; yt=y_true[:,j]; yp=y_pred[:,j]
        r2=r2_score(yt,yp)
        ax.scatter(yt,yp,s=14,alpha=0.6,edgecolors="none",color="#1f4e79")
        lo=min(yt.min(),yp.min()); hi=max(yt.max(),yp.max())
        pad=0.05*(hi-lo+1e-9)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=1)
        ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(lo-pad,hi+pad)
        ax.set_title(nm.replace("_"," "))
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.text(0.05,0.92,"R$^2$ = %.3f"%r2,transform=ax.transAxes,
                va="top",ha="left")
    for k in range(n,len(axes)): axes[k].axis("off")
    fig.tight_layout()
    out=os.path.join(PLOT_DIR,"parity_%s.png"%arch_tag)
    fig.savefig(out,bbox_inches="tight"); plt.close(fig)
    return out

def plot_r2_bar(metrics_rows, arch_tag):
    """Bar chart of R2 per output for one architecture."""
    names=[r["output"].replace("_"," ") for r in metrics_rows]
    r2s=[r["R2"] for r in metrics_rows]
    fig,ax=plt.subplots(figsize=(max(7,len(names)*0.5),4.2))
    bars=ax.bar(range(len(names)),r2s,color="#2e6f95",width=0.7)
    ax.axhline(0.9,color="grey",ls=":",lw=1)
    ax.set_ylim(0,1.05); ax.set_ylabel("Test R$^2$")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=90)
    ax.set_title("ANN test-set R$^2$ per output (%s)"%arch_tag)
    for b,v in zip(bars,r2s):
        ax.text(b.get_x()+b.get_width()/2,v+0.01,"%.2f"%v,ha="center",va="bottom",fontsize=9)
    fig.tight_layout()
    out=os.path.join(PLOT_DIR,"r2_bar_%s.png"%arch_tag)
    fig.savefig(out,bbox_inches="tight"); plt.close(fig)
    return out

def log(msg, fh=None):
    print(msg)
    if fh: fh.write(msg+"\n"); fh.flush()

# ============================================================ LOAD =============
def load_data(summary):
    if not os.path.exists(DATASET_CSV):
        sys.exit("dataset not found: %s"%DATASET_CSV)
    df = pd.read_csv(DATASET_CSV)
    n_all = len(df)
    if "status" in df.columns:
        df = df[df["status"].isin(USE_STATUS)].reset_index(drop=True)
    log("loaded %d rows; using %d rows with status in %s"%(n_all,len(df),USE_STATUS), summary)
    # sanity: required columns present
    missing = [c for c in INPUT_COLS+OUTPUT_COLS if c not in df.columns]
    if missing:
        sys.exit("missing columns in CSV: %s"%missing[:6])
    X = df[INPUT_COLS].astype(float).values
    Y = df[OUTPUT_COLS].astype(float).values
    return X, Y

# ============================================================ MODEL ============
def build_mlp(n_in, n_out):
    m = Sequential([Input(shape=(n_in,))])
    for h in HIDDEN:
        m.add(Dense(h, activation=ACTIVATION))
    m.add(Dense(n_out, activation="linear"))
    m.compile(optimizer=Adam(LR), loss="mse")
    return m

def split(X, Y):
    # 70/15/15 with fixed seed: first carve test, then val from the remainder
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=TEST_FRAC, random_state=SEED)
    val_rel = VAL_FRAC/(1.0-TEST_FRAC)
    X_tr, X_va, Y_tr, Y_va = train_test_split(X_tr, Y_tr, test_size=val_rel, random_state=SEED)
    return X_tr, X_va, X_te, Y_tr, Y_va, Y_te

def metrics_block(y_true_phys, y_pred_phys, names):
    rows=[]
    for j,nm in enumerate(names):
        yt=y_true_phys[:,j]; yp=y_pred_phys[:,j]
        mse=mean_squared_error(yt,yp)
        rows.append(dict(output=nm, R2=r2_score(yt,yp), MSE=mse, RMSE=np.sqrt(mse)))
    return rows

# ============================================================ TRAIN ============
def main():
    summary=open(os.path.join(MODEL_DIR,"training_summary.txt"),"w")
    log("="*64, summary); log("STEP 2 - ANN training (Keras)", summary); log("="*64, summary)
    X, Y = load_data(summary)
    X_tr,X_va,X_te,Y_tr,Y_va,Y_te = split(X,Y)
    log("split: train=%d val=%d test=%d"%(len(X_tr),len(X_va),len(X_te)), summary)

    # scale inputs (shared by all models) on TRAIN only
    sx=StandardScaler().fit(X_tr)
    Xtr,Xva,Xte = sx.transform(X_tr),sx.transform(X_va),sx.transform(X_te)
    joblib.dump(sx, os.path.join(MODEL_DIR,"scaler_X.pkl"))

    all_metrics=[]
    es=lambda: EarlyStopping(monitor="val_loss",patience=PATIENCE,restore_best_weights=True)

    # ---------- (1) SINGLE 18-output network ----------
    log("\n--- Architecture 1: single 18-output network ---", summary)
    sy=StandardScaler().fit(Y_tr)
    Ytr,Yva = sy.transform(Y_tr),sy.transform(Y_va)
    joblib.dump(sy, os.path.join(MODEL_DIR,"scaler_y.pkl"))
    m=build_mlp(len(INPUT_COLS), len(OUTPUT_COLS))
    h=m.fit(Xtr,Ytr,validation_data=(Xva,Yva),epochs=MAX_EPOCHS,batch_size=BATCH,
            verbose=0,callbacks=[es()])
    m.save(os.path.join(MODEL_DIR,"single_18output.keras"))
    Yte_pred = sy.inverse_transform(m.predict(Xte,verbose=0))
    rows=metrics_block(Y_te, Yte_pred, OUTPUT_COLS)
    for r in rows: r["architecture"]="single"; all_metrics.append(r)
    log("  trained %d epochs; mean test R2 = %.4f"
        %(len(h.history["loss"]), np.mean([r["R2"] for r in rows])), summary)
    # publication plots for the single 18-output net
    p1=plot_parity(Y_te, Yte_pred, OUTPUT_COLS, "single18")
    p2=plot_r2_bar(rows, "single18")
    log("  plots: %s , %s"%(os.path.basename(p1),os.path.basename(p2)), summary)
    # collect per-section predictions to assemble a combined per-section parity later
    SEC_COLS=[c for s in SECTIONS for c in sec_outputs(s)]  # the 18 section outputs
    sec_idx=[OUTPUT_COLS.index(c) for c in SEC_COLS]
    persec_true=np.zeros((len(Y_te),len(SEC_COLS)))
    persec_pred=np.zeros((len(Y_te),len(SEC_COLS)))

    # ---------- (2) SIX per-section networks ----------
    log("\n--- Architecture 2: six per-section networks ---", summary)
    for s in SECTIONS:
        cols=sec_outputs(s); idx=[OUTPUT_COLS.index(c) for c in cols]
        ytr,yva,yte = Y_tr[:,idx],Y_va[:,idx],Y_te[:,idx]
        syi=StandardScaler().fit(ytr)
        joblib.dump(syi, os.path.join(MODEL_DIR,"scaler_y_%s.pkl"%s))
        mi=build_mlp(len(INPUT_COLS), 3)
        hi=mi.fit(Xtr,syi.transform(ytr),validation_data=(Xva,syi.transform(yva)),
                  epochs=MAX_EPOCHS,batch_size=BATCH,verbose=0,callbacks=[es()])
        mi.save(os.path.join(MODEL_DIR,"section_%s.keras"%s))
        yte_pred=syi.inverse_transform(mi.predict(Xte,verbose=0))
        rows=metrics_block(yte, yte_pred, cols)
        for r in rows: r["architecture"]="per_section"; all_metrics.append(r)
        # stash into combined arrays (in SEC_COLS order) for one big parity plot
        for k,c in enumerate(cols):
            ci=SEC_COLS.index(c)
            persec_true[:,ci]=yte[:,k]; persec_pred[:,ci]=yte_pred[:,k]
        log("  %-14s %d epochs; section mean R2 = %.4f"
            %(s,len(hi.history["loss"]),np.mean([r["R2"] for r in rows])), summary)

    # combined per-section plots (all 18 outputs from the six section nets)
    ps_rows=[r for r in all_metrics if r["architecture"]=="per_section"]
    # reorder ps_rows to OUTPUT_COLS order
    ps_by_name={r["output"]:r for r in ps_rows}
    ps_ordered=[ps_by_name[c] for c in SEC_COLS]
    pp=plot_parity(persec_true, persec_pred, SEC_COLS, "persection")
    pb=plot_r2_bar(ps_ordered, "persection")
    log("  plots: %s , %s"%(os.path.basename(pp),os.path.basename(pb)), summary)

    # ---------- report ----------
    md=pd.DataFrame(all_metrics)[["architecture","output","R2","MSE","RMSE"]]
    md.to_csv(os.path.join(MODEL_DIR,"metrics_report.csv"),index=False)
    log("\n=== TEST-SET ACCURACY (per output) ===", summary)
    for arch in ["single","per_section"]:
        sub=md[md.architecture==arch]
        log("  %-11s : mean R2=%.4f  median R2=%.4f  mean RMSE=%.4g"
            %(arch, sub.R2.mean(), sub.R2.median(), sub.RMSE.mean()), summary)
    log("\nsaved models + scalers + metrics_report.csv -> %s"%MODEL_DIR, summary)
    log("plots (parity + R2 bar, Times New Roman 12) -> %s"%PLOT_DIR, summary)
    log("Next (Step 3): load single_18output.keras + scaler_X/scaler_y for SHAP.", summary)
    summary.close()
    print("\nDONE. Artifacts in:", MODEL_DIR)

if __name__=="__main__":
    main()
