#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 4  -  PSO optimization (surrogate-based)
================================================================================
Uses the trained single 18-output ANN (Step 2) as a fast surrogate inside a
Particle Swarm Optimizer to find the operating point that MINIMIZES total plant
irreversibility (sum of the six section irreversibilities, I_B).

Constraints (as agreed):
  * Box bounds: every one of the 26 inputs is confined to the +-10% envelope on
    which the ANN was trained, so PSO never leaves the surrogate's valid region.
    (This also keeps the three split fractions physically valid automatically.)
  * Non-negativity penalty: any candidate whose predicted section irreversibility
    is negative receives a large penalty, so PSO cannot "win" by exploiting an
    unphysical corner of the surrogate.

Outputs (into <ASPEN_FOLDER>/ANN_models/pso/):
  pso_result.csv           - optimised 26 inputs, base vs optimum, with % change
  pso_section_summary.csv  - per-section I_B/eps/IP at base vs optimum
  pso_convergence.png      - best-objective vs iteration (Times New Roman 12)
  pso_summary.txt          - readable report

Self-contained PSO (no external optimizer dependency). Reproducible (fixed seed).

RUN:  pip install tensorflow scikit-learn pandas joblib matplotlib numpy
      python ann_pso_step4.py
Requires Step 2 artifacts in ANN_models/: single_18output.keras, scaler_X.pkl,
scaler_y.pkl, and the dataset CSV (to set the +-10% bounds and base case).
================================================================================
"""
import os, sys, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models")
PSO_DIR      = os.path.join(MODEL_DIR, "pso")

SEED        = 42
USE_STATUS  = ("OK",)
ENVELOPE    = 0.10          # +-10% box around base case (matches training)

# Base-case operating point the +-10% envelope is centred on. Edit these to your
# true validated base case so the optimum is reported relative to the SAME point
# used elsewhere in the paper (plant total irreversibility 108.3 MW). If left as
# None, the dataset mean is used instead.
BASE_CASE = {
 "rawmeal_feed[kg/s]":100.78, "rmgas_flow[kg/s]":178.90, "rmgas_temp[K]":436.15,
 "coal_feed[kg/s]":8.28, "cmgas_flow[kg/s]":33.60, "cmgas_temp[K]":436.15,
 "clrcair_flow[kg/s]":48.00, "clrcair2_flow[kg/s]":103.00,
 "rmfan_T[K]":438.15, "rmdry_T[K]":363.15, "gasht_T[K]":438.15, "cmdry_T[K]":343.15,
 "phk1_T[K]":593.15, "phk2_T[K]":753.15, "phk3_T[K]":903.15, "phk4_T[K]":973.15,
 "phk5_T[K]":1023.15, "phc1_T[K]":593.15, "phc2_T[K]":753.15, "phc3_T[K]":903.15,
 "phc4_T[K]":973.15, "phc5_T[K]":1023.15, "clcalc_T[K]":1173.15,
 "coal_split[-]":0.55, "secair_split[-]":0.55, "meal_split[-]":0.65,
}

# PSO hyperparameters (standard, robust values)
N_PARTICLES = 40
N_ITERS     = 200
W_INERTIA   = 0.7298        # Clerc-Kennedy constriction values
C1_COGN     = 1.4962
C2_SOCIAL   = 1.4962
PENALTY     = 1.0e6         # added per negative-irreversibility violation

INPUT_COLS = [
 "rawmeal_feed[kg/s]","rmgas_flow[kg/s]","rmgas_temp[K]","coal_feed[kg/s]",
 "cmgas_flow[kg/s]","cmgas_temp[K]","clrcair_flow[kg/s]","clrcair2_flow[kg/s]",
 "rmfan_T[K]","rmdry_T[K]","gasht_T[K]","cmdry_T[K]",
 "phk1_T[K]","phk2_T[K]","phk3_T[K]","phk4_T[K]","phk5_T[K]",
 "phc1_T[K]","phc2_T[K]","phc3_T[K]","phc4_T[K]","phc5_T[K]",
 "clcalc_T[K]","coal_split[-]","secair_split[-]","meal_split[-]",
]
SECTIONS = ["RawMill","CoalMill","Preheater","Calciner","Kiln","ClinkerCooler"]
def sec_outputs(s): return ["%s_I_B_MW"%s, "%s_eps"%s, "%s_IP_MW"%s]
OUTPUT_COLS = [c for s in SECTIONS for c in sec_outputs(s)] + ["clinker_ex_MW"]  # 19
IB_IDX  = [OUTPUT_COLS.index("%s_I_B_MW"%s) for s in SECTIONS]  # the six I_B columns
CLK_IDX = OUTPUT_COLS.index("clinker_ex_MW")                    # clinker exergy output

# ============================================================ SETUP ============
np.random.seed(SEED)
os.makedirs(PSO_DIR, exist_ok=True)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import load_model
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
def setup_fonts():
    for fam in ["Times New Roman","Times","DejaVu Serif"]:
        if any(f.name==fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]=fam; break
    else: plt.rcParams["font.family"]="serif"
    plt.rcParams.update({"font.size":12,"axes.titlesize":12,"axes.labelsize":12,
        "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
        "figure.dpi":120,"savefig.dpi":300,"axes.linewidth":0.8})
setup_fonts()

# ============================================================ LOAD =============
def load_all():
    for fp in ["single_18output.keras","scaler_X.pkl","scaler_y.pkl"]:
        if not os.path.exists(os.path.join(MODEL_DIR,fp)):
            sys.exit("missing Step-2 artifact: %s (run ann_train_step2.py first)"%fp)
    if not os.path.exists(DATASET_CSV): sys.exit("dataset not found: %s"%DATASET_CSV)
    model=load_model(os.path.join(MODEL_DIR,"single_18output.keras"))
    sx=joblib.load(os.path.join(MODEL_DIR,"scaler_X.pkl"))
    sy=joblib.load(os.path.join(MODEL_DIR,"scaler_y.pkl"))
    df=pd.read_csv(DATASET_CSV)
    if "status" in df.columns: df=df[df["status"].isin(USE_STATUS)].reset_index(drop=True)
    X=df[INPUT_COLS].astype(float).values
    return model,sx,sy,X

# ============================================================ SURROGATE ========
def make_predictor(model, sx, sy):
    """Vectorised: physical inputs (M,26) -> physical outputs (M,18)."""
    def predict(Xphys):
        Xs=sx.transform(Xphys)
        Ys=model.predict(Xs, verbose=0)
        return sy.inverse_transform(Ys)
    return predict

def objective(Xphys, predict):
    """SPECIFIC plant irreversibility = (sum of six I_B) / clinker product exergy.
       Minimising this rewards genuine thermodynamic efficiency rather than simply
       reducing throughput. Penalties applied for any negative section
       irreversibility and for a non-physical (<=0) clinker exergy.
       Xphys: (M,26) -> returns (M,) objective to MINIMIZE, and the raw outputs."""
    Y=predict(Xphys)                      # (M,19) physical
    ib=Y[:,IB_IDX]                        # (M,6)
    total=ib.sum(axis=1)                  # sum of six section irreversibilities (MW)
    clk=Y[:,CLK_IDX]                      # clinker product exergy (MW)
    clk_safe=np.where(clk>1e-6, clk, 1e-6)
    spec=total/clk_safe                   # specific irreversibility [MW/MW clinker]
    neg=np.clip(-ib,0,None).sum(axis=1)   # negative-irreversibility magnitude
    nviol=(ib<0).sum(axis=1)              # count of negative-I_B violations
    badclk=(clk<=1e-6).astype(float)      # non-physical clinker output
    return spec + PENALTY*nviol + PENALTY*neg + PENALTY*badclk, Y

# ============================================================ PSO ==============
def run_pso(predict, lb, ub, base_x):
    d=len(lb); rng=np.random.default_rng(SEED)
    # init swarm uniformly in the box; seed one particle at the base case
    Xp=rng.uniform(lb,ub,size=(N_PARTICLES,d))
    Xp[0]=base_x.copy()
    Vp=rng.uniform(-(ub-lb),(ub-lb),size=(N_PARTICLES,d))*0.1
    fval,_=objective(Xp,predict)
    pbest=Xp.copy(); pbest_f=fval.copy()
    g=int(np.argmin(pbest_f)); gbest=pbest[g].copy(); gbest_f=pbest_f[g]
    history=[gbest_f]
    for it in range(N_ITERS):
        r1=rng.random((N_PARTICLES,d)); r2=rng.random((N_PARTICLES,d))
        Vp=(W_INERTIA*Vp + C1_COGN*r1*(pbest-Xp) + C2_SOCIAL*r2*(gbest-Xp))
        Xp=Xp+Vp
        Xp=np.clip(Xp,lb,ub)              # enforce box bounds every step
        fval,_=objective(Xp,predict)
        improved=fval<pbest_f
        pbest[improved]=Xp[improved]; pbest_f[improved]=fval[improved]
        g=int(np.argmin(pbest_f))
        if pbest_f[g]<gbest_f: gbest=pbest[g].copy(); gbest_f=pbest_f[g]
        history.append(gbest_f)
    return gbest, gbest_f, np.array(history)

# ============================================================ MAIN =============
def main():
    model,sx,sy,X = load_all()
    predict=make_predictor(model,sx,sy)

    # base case = the validated operating point (BASE_CASE), else dataset mean
    if BASE_CASE is not None:
        base_x = np.array([BASE_CASE[c] for c in INPUT_COLS], dtype=float)
    else:
        base_x = X.mean(axis=0)
    lb = base_x*(1.0-ENVELOPE); ub = base_x*(1.0+ENVELOPE)
    # for the three split fractions and temperatures the *= envelope is fine since
    # all are positive; ensure lb<ub elementwise
    lb,ub=np.minimum(lb,ub),np.maximum(lb,ub)

    # base-case prediction
    base_pred=predict(base_x[None,:])[0]
    base_total=base_pred[IB_IDX].sum()
    base_clk=base_pred[CLK_IDX]
    base_spec=base_total/base_clk if base_clk>0 else float("nan")

    print(">> running PSO (%d particles x %d iters) to minimise SPECIFIC irreversibility..."
          %(N_PARTICLES,N_ITERS))
    gbest,gbest_f,hist=run_pso(predict,lb,ub,base_x)
    opt_pred=predict(gbest[None,:])[0]
    opt_total=opt_pred[IB_IDX].sum()
    opt_clk=opt_pred[CLK_IDX]
    opt_spec=opt_total/opt_clk if opt_clk>0 else float("nan")

    # ---- report inputs base vs optimum ----
    rows=[]
    for i,c in enumerate(INPUT_COLS):
        rows.append(dict(input=c, base=base_x[i], optimum=gbest[i],
                         pct_change=100.0*(gbest[i]-base_x[i])/base_x[i]))
    pd.DataFrame(rows).to_csv(os.path.join(PSO_DIR,"pso_result.csv"),index=False)

    # ---- report per-section outputs base vs optimum ----
    srows=[]
    for k,s in enumerate(SECTIONS):
        i_ib=OUTPUT_COLS.index("%s_I_B_MW"%s)
        i_ep=OUTPUT_COLS.index("%s_eps"%s)
        i_ip=OUTPUT_COLS.index("%s_IP_MW"%s)
        srows.append(dict(section=s,
            I_B_base=base_pred[i_ib], I_B_opt=opt_pred[i_ib],
            eps_base=base_pred[i_ep], eps_opt=opt_pred[i_ep],
            IP_base=base_pred[i_ip], IP_opt=opt_pred[i_ip]))
    sdf=pd.DataFrame(srows); sdf.to_csv(os.path.join(PSO_DIR,"pso_section_summary.csv"),index=False)

    # ---- convergence plot ----
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(range(len(hist)),hist,color="#1f4e79",lw=1.6)
    ax.set_xlabel("PSO iteration"); ax.set_ylabel("Total plant irreversibility (MW)")
    ax.set_title("PSO convergence — total irreversibility minimisation")
    fig.tight_layout(); fig.savefig(os.path.join(PSO_DIR,"pso_convergence.png"),
                                    bbox_inches="tight"); plt.close(fig)

    # ---- text summary ----
    red=base_total-opt_total; redpct=100.0*red/base_total if base_total else 0
    with open(os.path.join(PSO_DIR,"pso_summary.txt"),"w") as fh:
        def w(m): print(m); fh.write(m+"\n")
        w("="*64); w("STEP 4 - PSO surrogate optimisation"); w("="*64)
        w("objective : minimise sum of six section irreversibilities")
        w("surrogate : single_18output.keras")
        w("bounds    : +-%.0f%% box around base case (validated region)"%(ENVELOPE*100))
        w("swarm     : %d particles, %d iterations, seed %d"%(N_PARTICLES,N_ITERS,SEED))
        w("")
        w("objective: SPECIFIC irreversibility = total I_B / clinker exergy")
        w("")
        w("                                   base      optimum")
        w("total irreversibility   [MW]   = %8.3f   %8.3f"%(base_total,opt_total))
        w("clinker product exergy  [MW]   = %8.3f   %8.3f"%(base_clk,opt_clk))
        w("specific irreversibility[-]    = %8.4f   %8.4f"%(base_spec,opt_spec))
        spec_red=base_spec-opt_spec; spec_pct=100*spec_red/base_spec if base_spec else 0
        w("")
        w("specific-irreversibility reduction = %.4f  (%.1f%%)"%(spec_red,spec_pct))
        w("total-irreversibility change       = %.3f MW (%.1f%%)"%(red,redpct))
        w("")
        w("per-section irreversibility (MW): base -> optimum")
        for _,r in sdf.iterrows():
            w("  %-14s %7.3f -> %7.3f"%(r['section'],r['I_B_base'],r['I_B_opt']))
        w("")
        w("optimised operating point (26 inputs): see pso_result.csv")
        w("artifacts -> %s"%PSO_DIR)
    print("\nDONE. PSO artifacts in:", PSO_DIR)

if __name__=="__main__":
    main()
