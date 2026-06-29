#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 2 (HYBRID)  -  per-section model selection (ANN vs Random Forest)
================================================================================
Trains BOTH surrogates on the SAME train split, then for each section selects
the architecture that performs better on a held-out VALIDATION set, and reports
final accuracy on a separate TEST set that the selection never saw. This avoids
test-set leakage: selection uses validation, reporting uses test.

  * ANN  : single multi-output net (26 -> 21), tanh 64-32 (scaled I/O)
  * RF   : 21 dedicated per-output Random Forests (raw I/O)
  * route: per SECTION (its 3 metrics go to one model) chosen on validation;
           the 3 plant-level outputs are routed individually on validation.

Empirically RF wins the drying-dominated mills (raw/coal mill I_B ~0.94 vs the
ANN's ~0.67/0.79) while the ANN is competitive elsewhere, so the hybrid keeps
the best of both. Described in one sentence: "for each section the better of an
ANN and a Random Forest was selected on a validation set and evaluated on a
held-out test set."

Saves into  <ASPEN_FOLDER>/ANN_models/HYBRID/ :
  hybrid_ann.keras                 (the 21-output ANN)
  scaler_X.pkl, scaler_y.pkl       (ANN input/output scalers)
  hybrid_bundle.joblib             ({forests[21], routing[21], output_cols,
                                     input_cols, kind}) - plain, no custom class
  metrics_report.csv               (val + test R2 per output, both models, chosen)
  routing.csv                      (per output: chosen model + why)
  training_summary.txt, plots/

Step 3 (SHAP) / Step 4 (PSO) load this and predict via the helper documented at
the bottom of this file (ANN-routed outputs via the scaled ANN, RF-routed via
the raw forests).

RUN:  pip install tensorflow scikit-learn pandas joblib matplotlib
      python hybrid_train_step2.py
================================================================================
"""
import os, sys, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models", "HYBRID")

SEED       = 42
TEST_FRAC  = 0.15
VAL_FRAC   = 0.15
USE_STATUS = ("OK",)
DROP_BASE  = False

# ANN hyperparams
HIDDEN=(64,32); ACT="tanh"; LR=1e-3; BATCH=32; MAX_EPOCHS=500; PATIENCE=30
# RF hyperparams
N_EST=400; N_JOBS=-1

INPUT_COLS = [
 "rawmeal_feed[kg/s]","rmgas_flow[kg/s]","rmgas_temp[K]","coal_feed[kg/s]",
 "cmgas_flow[kg/s]","cmgas_temp[K]","clrcair_flow[kg/s]","clrcair2_flow[kg/s]",
 "rmfan_T[K]","rmdry_T[K]","gasht_T[K]","cmdry_T[K]",
 "phk1_T[K]","phk2_T[K]","phk3_T[K]","phk4_T[K]","phk5_T[K]",
 "phc1_T[K]","phc2_T[K]","phc3_T[K]","phc4_T[K]","phc5_T[K]",
 "clcalc_T[K]","coal_split[-]","secair_split[-]","meal_split[-]",
]
SECTIONS=["RawMill","CoalMill","Preheater","Calciner","Kiln","ClinkerCooler"]
def sec_outputs(s): return ["%s_I_B_MW"%s,"%s_eps"%s,"%s_IP_MW"%s]
SECTION_COLS=[c for s in SECTIONS for c in sec_outputs(s)]      # 18
PLANT_OUTPUTS=["clinker_ex_MW","plant_spec_I_MJkg","plant_eps"] # 3
OUTPUT_COLS=SECTION_COLS+PLANT_OUTPUTS                          # 21

# ============================================================ SETUP ============
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
PLOT_DIR=os.path.join(MODEL_DIR,"plots"); os.makedirs(PLOT_DIR,exist_ok=True)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
def _fonts():
    for fam in ["Times New Roman","Times","DejaVu Serif"]:
        if any(f.name==fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]=fam; break
    else: plt.rcParams["font.family"]="serif"
    plt.rcParams.update({"font.size":12,"axes.titlesize":12,"axes.labelsize":12,
        "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
        "figure.dpi":120,"savefig.dpi":300,"axes.linewidth":0.8})
_fonts()

def log(m,fh=None):
    print(m)
    if fh: fh.write(m+"\n"); fh.flush()

# ============================================================ LOAD =============
def load_data(summary):
    if not os.path.exists(DATASET_CSV): sys.exit("dataset not found: %s"%DATASET_CSV)
    df=pd.read_csv(DATASET_CSV); n=len(df)
    if "status" in df.columns: df=df[df["status"].isin(USE_STATUS)].reset_index(drop=True)
    if DROP_BASE and "sample" in df.columns:
        df=df[df["sample"].astype(str).str.lower()!="base"].reset_index(drop=True)
    log("loaded %d rows; using %d (status %s, drop_base=%s)"%(n,len(df),USE_STATUS,DROP_BASE),summary)
    miss=[c for c in INPUT_COLS+OUTPUT_COLS if c not in df.columns]
    if miss: sys.exit("missing columns: %s"%miss[:6])
    return df[INPUT_COLS].astype(float).values, df[OUTPUT_COLS].astype(float).values

def three_way_split(X,Y):
    # 70/15/15 train/val/test, fixed seed (same split feeds ANN and RF)
    X_tr,X_te,Y_tr,Y_te=train_test_split(X,Y,test_size=TEST_FRAC,random_state=SEED)
    vr=VAL_FRAC/(1.0-TEST_FRAC)
    X_tr,X_va,Y_tr,Y_va=train_test_split(X_tr,Y_tr,test_size=vr,random_state=SEED)
    return X_tr,X_va,X_te,Y_tr,Y_va,Y_te

# ============================================================ TRAIN ============
def main():
    summary=open(os.path.join(MODEL_DIR,"training_summary.txt"),"w")
    log("="*64,summary); log("STEP 2 (HYBRID) - per-section ANN/RF selection",summary); log("="*64,summary)
    X,Y=load_data(summary)
    X_tr,X_va,X_te,Y_tr,Y_va,Y_te=three_way_split(X,Y)
    log("split: train=%d val=%d test=%d"%(len(X_tr),len(X_va),len(X_te)),summary)

    # ---- ANN: single 21-output net (scaled I/O), trained on TRAIN, early-stop on VAL ----
    sx=StandardScaler().fit(X_tr); sy=StandardScaler().fit(Y_tr)
    Xtr,Xva,Xte=sx.transform(X_tr),sx.transform(X_va),sx.transform(X_te)
    Ytr,Yva=sy.transform(Y_tr),sy.transform(Y_va)
    ann=Sequential([Input(shape=(len(INPUT_COLS),))])
    for h in HIDDEN: ann.add(Dense(h,activation=ACT))
    ann.add(Dense(len(OUTPUT_COLS),activation="linear"))
    ann.compile(optimizer=Adam(LR),loss="mse")
    es=EarlyStopping(monitor="val_loss",patience=PATIENCE,restore_best_weights=True)
    ann.fit(Xtr,Ytr,validation_data=(Xva,Yva),epochs=MAX_EPOCHS,batch_size=BATCH,verbose=0,callbacks=[es])
    ann.save(os.path.join(MODEL_DIR,"hybrid_ann.keras"))
    joblib.dump(sx,os.path.join(MODEL_DIR,"scaler_X.pkl")); joblib.dump(sy,os.path.join(MODEL_DIR,"scaler_y.pkl"))
    ann_val =sy.inverse_transform(ann.predict(Xva,verbose=0))
    ann_test=sy.inverse_transform(ann.predict(Xte,verbose=0))

    # ---- RF: 21 dedicated forests (raw I/O), trained on TRAIN ----
    forests=[]; rf_val=np.zeros_like(ann_val); rf_test=np.zeros_like(ann_test)
    for j in range(len(OUTPUT_COLS)):
        f=RandomForestRegressor(n_estimators=N_EST,n_jobs=N_JOBS,random_state=SEED).fit(X_tr,Y_tr[:,j])
        forests.append(f); rf_val[:,j]=f.predict(X_va); rf_test[:,j]=f.predict(X_te)

    # ---- SELECTION on VALIDATION: per section (mean over its 3 metrics) -------
    def r2col(yt,yp,j): return r2_score(yt[:,j],yp[:,j])
    routing={}                      # output_col -> 'ann' or 'rf'
    log("\n--- model selection on VALIDATION (per section) ---",summary)
    for s in SECTIONS:
        idx=[OUTPUT_COLS.index(c) for c in sec_outputs(s)]
        a=np.mean([r2col(Y_va,ann_val,j) for j in idx])
        r=np.mean([r2col(Y_va,rf_val,j)  for j in idx])
        win="rf" if r>a else "ann"
        for c in sec_outputs(s): routing[c]=win
        log("  %-14s val R2  ANN=%.3f  RF=%.3f  -> %s"%(s,a,r,win.upper()),summary)
    log("\n--- model selection on VALIDATION (plant outputs, individual) ---",summary)
    for c in PLANT_OUTPUTS:
        j=OUTPUT_COLS.index(c)
        a=r2col(Y_va,ann_val,j); r=r2col(Y_va,rf_val,j)
        win="rf" if r>a else "ann"; routing[c]=win
        log("  %-18s val R2  ANN=%.3f  RF=%.3f  -> %s"%(c,a,r,win.upper()),summary)

    # ---- assemble HYBRID predictions on TEST (selection-blind) ---------------
    route=[routing[c] for c in OUTPUT_COLS]
    hyb_test=np.where(np.array([rr=="ann" for rr in route])[None,:], ann_test, rf_test)

    # ---- final metrics on TEST -----------------------------------------------
    rows=[]
    for j,c in enumerate(OUTPUT_COLS):
        yt=Y_te[:,j]
        for tag,pred in [("ann",ann_test),("rf",rf_test),("hybrid",hyb_test)]:
            mse=mean_squared_error(yt,pred[:,j])
            rows.append(dict(output=c,model=tag,R2=r2_score(yt,pred[:,j]),RMSE=np.sqrt(mse),
                             chosen=(tag==routing[c]) if tag in ("ann","rf") else (tag=="hybrid")))
    md=pd.DataFrame(rows); md.to_csv(os.path.join(MODEL_DIR,"metrics_report.csv"),index=False)

    # routing table
    pd.DataFrame([{"output":c,"chosen_model":routing[c]} for c in OUTPUT_COLS]
                 ).to_csv(os.path.join(MODEL_DIR,"routing.csv"),index=False)

    # ---- save the hybrid bundle (plain; no custom class) ---------------------
    joblib.dump({"forests":forests,"routing":route,"output_cols":list(OUTPUT_COLS),
                 "input_cols":list(INPUT_COLS),"ann_file":"hybrid_ann.keras","kind":"hybrid_ann_rf"},
                os.path.join(MODEL_DIR,"hybrid_bundle.joblib"))

    # ---- report --------------------------------------------------------------
    def meanR2(model):
        sub=md[md.model==model]; return sub.R2.mean()
    log("\n=== HELD-OUT TEST ACCURACY (mean R2 over 21 outputs) ===",summary)
    log("  ANN-only    : %.4f"%meanR2("ann"),summary)
    log("  RF-only     : %.4f"%meanR2("rf"),summary)
    log("  HYBRID      : %.4f   <-- per-section best, selected on validation"%meanR2("hybrid"),summary)
    log("\nper-output TEST R2 (chosen model in CAPS):",summary)
    for j,c in enumerate(OUTPUT_COLS):
        a=md[(md.output==c)&(md.model=="ann")].R2.values[0]
        r=md[(md.output==c)&(md.model=="rf")].R2.values[0]
        ch=routing[c]
        mk=lambda tag,v: ("[%.3f]"%v if tag==ch else " %.3f "%v)
        log("  %-20s ANN%s RF%s -> %s"%(c,mk("ann",a),mk("rf",r),ch.upper()),summary)

    # ---- parity plot for the hybrid (section metrics) ------------------------
    fig,axes=plt.subplots(6,3,figsize=(3*3.2,6*3.0)); axes=np.array(axes).reshape(-1)
    for j,c in enumerate(SECTION_COLS):
        k=OUTPUT_COLS.index(c); ax=axes[j]; yt=Y_te[:,k]; yp=hyb_test[:,k]
        ax.scatter(yt,yp,s=14,alpha=0.6,edgecolors="none",color="#1f4e79")
        lo=min(yt.min(),yp.min()); hi=max(yt.max(),yp.max()); pad=0.05*(hi-lo+1e-9)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=1)
        ax.set_title("%s (%s)"%(c.replace("_"," "),routing[c].upper()))
        ax.text(0.05,0.92,"R$^2$=%.3f"%r2_score(yt,yp),transform=ax.transAxes,va="top")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR,"parity_hybrid_sections.png"),bbox_inches="tight"); plt.close(fig)

    log("\nsaved hybrid surrogate -> %s"%MODEL_DIR,summary)
    log("Step 3/4: load hybrid_bundle.joblib + hybrid_ann.keras + scaler_X/scaler_y;",summary)
    log("predict per the helper in this file's docstring (ANN-routed scaled, RF-routed raw).",summary)
    summary.close()
    print("\nDONE. Hybrid artifacts in:", MODEL_DIR)

# ---- reference predictor for Step 3/4 (copy into those scripts) --------------
def hybrid_predict(X_raw, MODEL_DIR=MODEL_DIR):
    """Load the hybrid and predict (n,21) in OUTPUT_COLS order from RAW inputs."""
    from tensorflow.keras.models import load_model
    d=joblib.load(os.path.join(MODEL_DIR,"hybrid_bundle.joblib"))
    sx=joblib.load(os.path.join(MODEL_DIR,"scaler_X.pkl")); sy=joblib.load(os.path.join(MODEL_DIR,"scaler_y.pkl"))
    ann=load_model(os.path.join(MODEL_DIR,d["ann_file"]))
    X_raw=np.asarray(X_raw,dtype=float)
    ann_all=sy.inverse_transform(ann.predict(sx.transform(X_raw),verbose=0))
    rf_all=np.column_stack([m.predict(X_raw) for m in d["forests"]])
    route=np.array([rr=="ann" for rr in d["routing"]])
    return np.where(route[None,:], ann_all, rf_all)

if __name__=="__main__":
    main()
