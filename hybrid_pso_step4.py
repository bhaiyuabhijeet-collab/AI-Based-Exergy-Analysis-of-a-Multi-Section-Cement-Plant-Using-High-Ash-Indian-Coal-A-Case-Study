#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 4 (+VALIDATION)  -  PSO optimisation  THEN  real-Aspen validation
================================================================================
ONE-STEP pipeline:

  PHASE 1 (surrogate)  - Particle Swarm Optimisation on the trained single
                         21-output ANN (18 section metrics + clinker exergy + plant specific-I + efficiency).
                         Minimises plant SPECIFIC irreversibility:
                             (sum of six section I_B) / clinker product exergy
                         over the +-10% box the ANN was trained on. Rewards
                         genuine efficiency gains, not throughput cuts.

  PHASE 2 (real model) - takes the 26 optimised inputs PSO found and evaluates
                         BOTH the untouched base case and the optimum in the
                         actual Aspen Plus flowsheet, through the same exergy
                         engine used everywhere else. These are the REAL numbers
                         for Table 6; the Phase-1 numbers live in surrogate space.

So a single run yields: the optimum operating point (Phase 1) AND its validated
thermodynamic effect (Phase 2), with no stale-CSV hand-off in between.

Outputs (into <ASPEN_FOLDER>/ANN_models/pso/):
  pso_result.csv           - optimised 26 inputs, base vs optimum, % change
  pso_section_summary.csv  - per-section I_B/eps/IP at base vs optimum (SURROGATE)
  pso_convergence.png      - best-objective vs iteration (Phase 1)
  pso_summary.txt          - readable PSO report (surrogate space)
  validate_optimum.csv     - REAL Aspen base vs optimum  <-- USE THIS FOR TABLE 6
  validate_optimum.txt     - readable validated report (real model)
  pso_section_bars.png/.pdf- per-section I_B base vs optimum (VALIDATED, Phase 2)
  pso_waterfall.png/.pdf   - waterfall of per-section contributions to the net
                             plant-irreversibility change (VALIDATED, Phase 2)

RUN:  python ann_pso_step4.py
Requires: Step-2 HYBRID artifacts (hybrid_ann.keras, scaler_X.pkl, scaler_y.pkl,
          the dataset CSV, AND for Phase 2: exergy_calculator.py,
          exergy_campaign_driver.py, the .bkp model, and pywin32 (Windows+Aspen).
          If Aspen/COM is unavailable, Phase 1 still completes and Phase 2 is
          skipped with a clear message.
================================================================================
"""
import os, sys, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models", "HYBRID")  # hybrid surrogate
PSO_DIR      = os.path.join(MODEL_DIR, "pso")
MODEL_FILE   = os.path.join(ASPEN_FOLDER, "new cmill rmill preheater calciner kiln cooler.bkp")

SEED        = 42
USE_STATUS  = ("OK",)
ENVELOPE    = 0.10          # +-10% box around base case (matches training)
RUN_VALIDATION = True       # set False to run PSO only (skip Aspen Phase 2)
SETTLE_RUNS = 2             # Aspen solve cadence (overridden by DRV.SETTLE_RUNS if present)

# ---- FIX_THROUGHPUT mode -----------------------------------------------------
# When True, the optimiser is forced to improve efficiency AT (essentially)
# CONSTANT PRODUCTION, so it cannot lower specific irreversibility by simply
# making more clinker (the throughput trick that gave the unphysical 79%).
#   * FIXED_INPUTS are pinned at their base values and removed from the swarm.
#   * a soft penalty holds predicted clinker exergy within +-CLK_BAND of base,
#     so the remaining free inputs cannot drift production either.
# Pin raw-meal feed (it sets the solid throughput -> clinker). Add
# "coal_feed[kg/s]" to also fix firing rate if you want a stricter constant-duty
# optimum.
FIX_THROUGHPUT = True
FIXED_INPUTS   = ["rawmeal_feed[kg/s]", "coal_feed[kg/s]",
                  "meal_split[-]", "coal_split[-]", "clcalc_T[K]"]
CLK_BAND       = 0.02       # allowed +-2% clinker drift in FIX_THROUGHPUT mode

# Base-case operating point the +-10% envelope is centred on (the validated point
# used everywhere else in the paper: plant total irreversibility 110.5 MW (write-then-solve),
# 35.9% efficiency, motor-inclusive). If left None, the dataset mean is used.
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

# PSO hyperparameters (Clerc-Kennedy constriction)
N_PARTICLES = 40
N_ITERS     = 200
W_INERTIA   = 0.7298
C1_COGN     = 1.4962
C2_SOCIAL   = 1.4962
PENALTY     = 1.0e6
BOUND_TOL   = 0.005         # flag inputs within 0.5% of a box bound

INPUT_COLS = [
 "rawmeal_feed[kg/s]","rmgas_flow[kg/s]","rmgas_temp[K]","coal_feed[kg/s]",
 "cmgas_flow[kg/s]","cmgas_temp[K]","clrcair_flow[kg/s]","clrcair2_flow[kg/s]",
 "rmfan_T[K]","rmdry_T[K]","gasht_T[K]","cmdry_T[K]",
 "phk1_T[K]","phk2_T[K]","phk3_T[K]","phk4_T[K]","phk5_T[K]",
 "phc1_T[K]","phc2_T[K]","phc3_T[K]","phc4_T[K]","phc5_T[K]",
 "clcalc_T[K]","coal_split[-]","secair_split[-]","meal_split[-]",
]
# bare driver keys for writing into Aspen (col-with-units -> DRV.set_input key)
COL2KEY = {
 "rawmeal_feed[kg/s]":"rawmeal_feed","rmgas_flow[kg/s]":"rmgas_flow","rmgas_temp[K]":"rmgas_temp",
 "coal_feed[kg/s]":"coal_feed","cmgas_flow[kg/s]":"cmgas_flow","cmgas_temp[K]":"cmgas_temp",
 "clrcair_flow[kg/s]":"clrcair_flow","clrcair2_flow[kg/s]":"clrcair2_flow",
 "rmfan_T[K]":"rmfan_T","rmdry_T[K]":"rmdry_T","gasht_T[K]":"gasht_T","cmdry_T[K]":"cmdry_T",
 "phk1_T[K]":"phk1_T","phk2_T[K]":"phk2_T","phk3_T[K]":"phk3_T","phk4_T[K]":"phk4_T","phk5_T[K]":"phk5_T",
 "phc1_T[K]":"phc1_T","phc2_T[K]":"phc2_T","phc3_T[K]":"phc3_T","phc4_T[K]":"phc4_T","phc5_T[K]":"phc5_T",
 "clcalc_T[K]":"clcalc_T","coal_split[-]":"coal_split","secair_split[-]":"secair_split","meal_split[-]":"meal_split",
}
SECTIONS  = ["RawMill","CoalMill","Preheater","Calciner","Kiln","ClinkerCooler"]   # output-column names
SEC_DISP  = ["Raw Mill","Coal Mill","Preheater","Calciner","Kiln","Clinker Cooler"] # DRV.evaluate keys
SEC_NICE  = ["Raw mill","Coal mill","Preheater","Calciner","Kiln","Clinker cooler"] # figure axis labels
def sec_outputs(s): return ["%s_I_B_MW"%s, "%s_eps"%s, "%s_IP_MW"%s]
# Must match the trained net's 21-output order EXACTLY (Step 2):
#   18 section metrics, clinker_ex_MW, plant_spec_I_MJkg, plant_eps.
OUTPUT_COLS = [c for s in SECTIONS for c in sec_outputs(s)] + \
              ["clinker_ex_MW", "plant_spec_I_MJkg", "plant_eps"]  # 21
IB_IDX  = [OUTPUT_COLS.index("%s_I_B_MW"%s) for s in SECTIONS]
CLK_IDX = OUTPUT_COLS.index("clinker_ex_MW")
SPEC_IDX = OUTPUT_COLS.index("plant_spec_I_MJkg")   # surrogate's own specific-I output
MW = 1e6

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

# ============================================================ FIGURES ==========
# Validated-result figures (Aspen base vs optimum). Drawn in Phase 2 from the
# section-level irreversibilities already computed there, so figure and Table 6
# come from the SAME validated numbers. Saved as PNG + PDF in PSO_DIR.
_BLUE,_GREEN,_RED,_GREY = "#2e6f95","#4f8a2e","#b5471d","#9aa0a6"

def plot_section_bars(labels, base, opt, fname="pso_section_bars.png"):
    """Grouped bars: per-section irreversibility, base vs optimum (MW)."""
    x=np.arange(len(labels)); w=0.38
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    ax.bar(x-w/2,base,w,label="Base",color=_BLUE)
    ax.bar(x+w/2,opt ,w,label="Optimum",color=_GREEN)
    for xi,(b,o) in enumerate(zip(base,opt)):
        ax.text(xi-w/2,b+0.4,"%.1f"%b,ha="center",va="bottom",fontsize=10)
        ax.text(xi+w/2,o+0.4,"%.1f"%o,ha="center",va="bottom",fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("Irreversibility (MW)"); ax.legend(frameon=False)
    ax.set_ylim(0,max(list(base)+list(opt))*1.12)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    for e in ("png","pdf"):
        fig.savefig(os.path.join(PSO_DIR,fname.replace(".png","."+e)),bbox_inches="tight")
    plt.close(fig)

def plot_waterfall(labels, base, opt, bt, ot, fname="pso_waterfall.png"):
    """Waterfall from base total (bt) to optimum total (ot) via signed per-section
       contributions (optimum - base). Green = reduction, red = increase."""
    from matplotlib.patches import Patch
    deltas=[o-b for b,o in zip(base,opt)]
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    ax.bar(0,bt,color=_GREY,width=0.6)
    ax.text(0,bt+0.3,"%.1f"%bt,ha="center",va="bottom",fontsize=11)
    running=bt
    for i,d in enumerate(deltas,start=1):
        bottom=running+min(d,0.0)
        ax.bar(i,abs(d),bottom=bottom,width=0.6,color=_GREEN if d<0 else _RED)
        ax.text(i,bottom+abs(d)+0.3,"%+.2f"%d,ha="center",va="bottom",fontsize=10)
        ax.plot([i-1+0.3,i-0.3],[running,running],color="grey",lw=0.8,ls=":")
        running+=d
    ax.bar(len(deltas)+1,ot,color=_GREY,width=0.6)
    ax.text(len(deltas)+1,ot+0.3,"%.1f"%ot,ha="center",va="bottom",fontsize=11)
    alllab=["Base"]+list(labels)+["Optimum"]
    ax.set_xticks(range(len(alllab))); ax.set_xticklabels(alllab,rotation=20,ha="right")
    ax.set_ylabel("Plant irreversibility (MW)")
    ax.set_ylim(max(0.0,min(ot,bt)-6.5), bt*1.04)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(handles=[Patch(color=_GREEN,label="Reduction"),Patch(color=_RED,label="Increase")],
              frameon=False,loc="upper right")
    fig.tight_layout()
    for e in ("png","pdf"):
        fig.savefig(os.path.join(PSO_DIR,fname.replace(".png","."+e)),bbox_inches="tight")
    plt.close(fig)

# ============================================================ LOAD =============
def load_all():
    need=["hybrid_ann.keras","scaler_X.pkl","scaler_y.pkl","hybrid_bundle.joblib"]
    for fp in need:
        if not os.path.exists(os.path.join(MODEL_DIR,fp)):
            sys.exit("missing Step-2 hybrid artifact: %s (run hybrid_train_step2.py first)"%fp)
    if not os.path.exists(DATASET_CSV): sys.exit("dataset not found: %s"%DATASET_CSV)
    ann=load_model(os.path.join(MODEL_DIR,"hybrid_ann.keras"))
    sx=joblib.load(os.path.join(MODEL_DIR,"scaler_X.pkl"))
    sy=joblib.load(os.path.join(MODEL_DIR,"scaler_y.pkl"))
    bundle=joblib.load(os.path.join(MODEL_DIR,"hybrid_bundle.joblib"))
    # sanity: the hybrid's output order must match this script's OUTPUT_COLS
    if list(bundle["output_cols"])!=list(OUTPUT_COLS):
        sys.exit("hybrid output order != PSO OUTPUT_COLS; re-check Step-2/Step-4 column lists.")
    df=pd.read_csv(DATASET_CSV)
    if "status" in df.columns: df=df[df["status"].isin(USE_STATUS)].reset_index(drop=True)
    X=df[INPUT_COLS].astype(float).values
    return (ann,sx,sy,bundle),X

# ============================================================ SURROGATE ========
# clinker target for FIX_THROUGHPUT mode (set in phase1; None disables the band)
_CLK_TARGET = [None]
_CLK_TOL    = [CLK_BAND]

def make_predictor(model_obj, sx, sy):
    # model_obj is the hybrid tuple (ann, sx, sy, bundle). Predict each output with
    # ITS routed model: ANN (scaled I/O) for ANN-routed outputs, the dedicated RF
    # forest (raw I/O) for RF-routed outputs. Returns physical (n,21) in OUTPUT_COLS.
    ann, _sx, _sy, bundle = model_obj
    forests=bundle["forests"]; route=np.array([r=="ann" for r in bundle["routing"]])
    def predict(Xphys):
        Xphys=np.asarray(Xphys,dtype=float)
        ann_all=sy.inverse_transform(ann.predict(sx.transform(Xphys), verbose=0))   # (n,21)
        rf_all=np.column_stack([m.predict(Xphys) for m in forests])                  # (n,21)
        return np.where(route[None,:], ann_all, rf_all)
    return predict

def objective(Xphys, predict):
    """SPECIFIC plant irreversibility = (sum of six I_B) / clinker product exergy.
       Penalised for negative section irreversibility and non-physical clinker.
       In FIX_THROUGHPUT mode a soft band penalty also holds clinker near base,
       so the optimum cannot lower specific-I by inflating production."""
    Y=predict(Xphys)
    ib=Y[:,IB_IDX]
    total=ib.sum(axis=1)
    clk=Y[:,CLK_IDX]
    clk_safe=np.where(clk>1e-6, clk, 1e-6)
    spec=total/clk_safe
    neg=np.clip(-ib,0,None).sum(axis=1)
    nviol=(ib<0).sum(axis=1)
    badclk=(clk<=1e-6).astype(float)
    pen = PENALTY*nviol + PENALTY*neg + PENALTY*badclk
    if _CLK_TARGET[0] is not None:
        dev = np.abs(clk - _CLK_TARGET[0])/_CLK_TARGET[0]    # fractional clinker drift
        band = np.clip(dev - _CLK_TOL[0], 0.0, None)         # 0 inside band, grows outside
        pen = pen + PENALTY*band
    return spec + pen, Y

# ============================================================ PSO ==============
def run_pso(predict, lb, ub, base_x):
    d=len(lb); rng=np.random.default_rng(SEED)
    Xp=rng.uniform(lb,ub,size=(N_PARTICLES,d)); Xp[0]=base_x.copy()
    Vp=rng.uniform(-(ub-lb),(ub-lb),size=(N_PARTICLES,d))*0.1
    fval,_=objective(Xp,predict)
    pbest=Xp.copy(); pbest_f=fval.copy()
    g=int(np.argmin(pbest_f)); gbest=pbest[g].copy(); gbest_f=pbest_f[g]
    history=[gbest_f]
    for it in range(N_ITERS):
        r1=rng.random((N_PARTICLES,d)); r2=rng.random((N_PARTICLES,d))
        Vp=(W_INERTIA*Vp + C1_COGN*r1*(pbest-Xp) + C2_SOCIAL*r2*(gbest-Xp))
        Xp=np.clip(Xp+Vp,lb,ub)
        fval,_=objective(Xp,predict)
        improved=fval<pbest_f
        pbest[improved]=Xp[improved]; pbest_f[improved]=fval[improved]
        g=int(np.argmin(pbest_f))
        if pbest_f[g]<gbest_f: gbest=pbest[g].copy(); gbest_f=pbest_f[g]
        history.append(gbest_f)
    return gbest, gbest_f, np.array(history)

def bound_report(gbest, lb, ub):
    """Flag any optimised input pinned at a box bound (possible throughput trick).
       Inputs intentionally held in FIX_THROUGHPUT mode are not flagged."""
    fixed = set(FIXED_INPUTS) if FIX_THROUGHPUT else set()
    flagged=[]
    for i,c in enumerate(INPUT_COLS):
        if c in fixed: continue
        span=ub[i]-lb[i]
        if span<=0: continue
        if (gbest[i]-lb[i])/span < BOUND_TOL: flagged.append((c,"lower"))
        elif (ub[i]-gbest[i])/span < BOUND_TOL: flagged.append((c,"upper"))
    return flagged

# ============================================================ PHASE 1 ==========
def phase1_pso():
    model_obj,X = load_all()           # model_obj = (ann, sx, sy, bundle)
    _ann,sx,sy,_bundle = model_obj
    predict=make_predictor(model_obj,sx,sy)

    if BASE_CASE is not None:
        base_x = np.array([BASE_CASE[c] for c in INPUT_COLS], dtype=float)
    else:
        base_x = X.mean(axis=0)
    lb = base_x*(1.0-ENVELOPE); ub = base_x*(1.0+ENVELOPE)
    lb,ub=np.minimum(lb,ub),np.maximum(lb,ub)

    base_pred=predict(base_x[None,:])[0]
    base_total=base_pred[IB_IDX].sum(); base_clk=base_pred[CLK_IDX]
    base_spec=base_total/base_clk if base_clk>0 else float("nan")

    # ---- FIX_THROUGHPUT: pin chosen inputs at base + hold clinker near base ----
    if FIX_THROUGHPUT:
        for c in FIXED_INPUTS:
            i=INPUT_COLS.index(c)
            lb[i]=ub[i]=base_x[i]              # pin: zero-width bound keeps it at base
        _CLK_TARGET[0]=float(base_clk)         # band-penalise clinker drift in objective
        _CLK_TOL[0]=CLK_BAND
        print(">> FIX_THROUGHPUT on: pinned %s ; clinker held within +-%.0f%% of base (%.2f MW)"
              %(", ".join(FIXED_INPUTS), CLK_BAND*100, base_clk))
    else:
        _CLK_TARGET[0]=None

    print(">> PHASE 1: PSO (%d particles x %d iters) minimising SPECIFIC irreversibility..."
          %(N_PARTICLES,N_ITERS))
    gbest,gbest_f,hist=run_pso(predict,lb,ub,base_x)
    opt_pred=predict(gbest[None,:])[0]
    opt_total=opt_pred[IB_IDX].sum(); opt_clk=opt_pred[CLK_IDX]
    opt_spec=opt_total/opt_clk if opt_clk>0 else float("nan")

    # inputs base vs optimum
    rows=[dict(input=c, base=base_x[i], optimum=gbest[i],
               pct_change=100.0*(gbest[i]-base_x[i])/base_x[i])
          for i,c in enumerate(INPUT_COLS)]
    pd.DataFrame(rows).to_csv(os.path.join(PSO_DIR,"pso_result.csv"),index=False)

    # per-section (surrogate)
    srows=[]
    for s in SECTIONS:
        i_ib=OUTPUT_COLS.index("%s_I_B_MW"%s); i_ep=OUTPUT_COLS.index("%s_eps"%s); i_ip=OUTPUT_COLS.index("%s_IP_MW"%s)
        srows.append(dict(section=s, I_B_base=base_pred[i_ib], I_B_opt=opt_pred[i_ib],
            eps_base=base_pred[i_ep], eps_opt=opt_pred[i_ep],
            IP_base=base_pred[i_ip], IP_opt=opt_pred[i_ip]))
    sdf=pd.DataFrame(srows); sdf.to_csv(os.path.join(PSO_DIR,"pso_section_summary.csv"),index=False)

    # convergence plot
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(range(len(hist)),hist,color="#1f4e79",lw=1.6)
    ax.set_xlabel("PSO iteration"); ax.set_ylabel("Specific irreversibility [MW/MW clinker]")
    ax.set_title("PSO convergence - specific-irreversibility minimisation")
    fig.tight_layout(); fig.savefig(os.path.join(PSO_DIR,"pso_convergence.png"),bbox_inches="tight"); plt.close(fig)

    # bound-pinning check
    flagged=bound_report(gbest,lb,ub)

    # text summary (surrogate space)
    with open(os.path.join(PSO_DIR,"pso_summary.txt"),"w") as fh:
        def w(m): print(m); fh.write(m+"\n")
        w("="*64); w("STEP 4 PHASE 1 - PSO surrogate optimisation"); w("="*64)
        w("objective : minimise SPECIFIC irreversibility (sum six I_B / clinker exergy)")
        w("surrogate : HYBRID (per-section ANN/RF; mills=RF, thermal=ANN), 21 outputs")
        w("bounds    : +-%.0f%% box around base case"%(ENVELOPE*100))
        if FIX_THROUGHPUT:
            w("mode      : FIX_THROUGHPUT - pinned %s; clinker held within +-%.0f%% of base"
              %(", ".join(FIXED_INPUTS), CLK_BAND*100))
        w("swarm     : %d particles, %d iters, seed %d"%(N_PARTICLES,N_ITERS,SEED))
        w("")
        w("                                   base      optimum   (SURROGATE)")
        w("total irreversibility   [MW]   = %8.3f   %8.3f"%(base_total,opt_total))
        w("clinker product exergy  [MW]   = %8.3f   %8.3f"%(base_clk,opt_clk))
        w("specific irreversibility[-]    = %8.4f   %8.4f"%(base_spec,opt_spec))
        sred=base_spec-opt_spec; spct=100*sred/base_spec if base_spec else 0
        w(""); w("specific-irreversibility reduction = %.4f (%.1f%%)"%(sred,spct))
        w("")
        if flagged:
            w("[bound check] inputs pinned at a box bound (review for throughput tricks):")
            for c,which in flagged: w("   %-18s -> %s bound"%(c,which))
            w("   NB: raw-meal/coal feed pinned at the UPPER bound can inflate clinker")
            w("       exergy and lower specific-I without a true efficiency gain.")
        else:
            w("[bound check] none - optimum is interior to the box (good).")
        w(""); w("optimised operating point: pso_result.csv")
    print(">> PHASE 1 done. Surrogate artifacts in:", PSO_DIR)
    return base_x, gbest, flagged

# ============================================================ PHASE 2 ==========
def phase2_validate(base_x, gbest, flagged):
    """Evaluate base (untouched) and optimum (write-then-solve) in REAL Aspen."""
    try:
        sys.path.insert(0, ASPEN_FOLDER)
        import exergy_calculator as EX      # noqa: F401  (used via DRV.evaluate)
        import exergy_campaign_driver as DRV
        import win32com.client as win32
    except Exception as e:
        print("\n>> PHASE 2 skipped (Aspen/COM not available here): %s"%str(e)[:120])
        print("   Phase-1 surrogate results are saved; run on the Aspen machine to validate.")
        return

    bkp = getattr(DRV, "BKP", MODEL_FILE)
    settle = getattr(DRV, "SETTLE_RUNS", SETTLE_RUNS)
    if not os.path.exists(bkp):
        print("\n>> PHASE 2 skipped: model .bkp not found: %s"%bkp); return

    def open_aspen():
        try: asp=win32.Dispatch("Apwn.Document.40.0")
        except Exception: asp=win32.Dispatch("Apwn.Document")
        try: asp.SuppressDialogs=1
        except Exception: pass
        asp.InitFromArchive2(os.path.abspath(bkp))
        try: asp.Visible=0
        except Exception: pass
        return asp

    def solve(asp):
        for _ in range(max(1,settle)): asp.Engine.Run2()

    def evaluate_case(asp, inputs=None):
        asp.Reinit()
        if inputs is not None:
            for col,val in inputs.items():
                DRV.set_input(asp, COL2KEY[col], float(val))
        solve(asp)
        return DRV.evaluate(asp)

    def summarise(sections, plant):
        total=plant["I_B"]/MW
        clk=plant.get("clk",0.0)/MW
        spec=total/clk if clk>0 else float("nan")
        eps=plant.get("eps",0.0)*100.0
        # DIAGNOSTIC: expose the efficiency denominator components so we can see
        # exactly which term moves between base and optimum (fuel vs electrical).
        fuel=plant.get("fuel_ex_fixed",float("nan"))/MW
        elec=plant.get("elec",plant.get("W_elec",float("nan")))
        try: elec=elec/MW
        except Exception: pass
        mcoal=plant.get("m_coal",float("nan"))
        print("      [diag] m_coal=%.4f kg/s  fuel_ex_fixed=%.3f MW  elec=%.3f MW  clk=%.3f MW  eff=%.1f%%"
              %(mcoal, fuel, elec if elec==elec else float('nan'), clk, eps))
        secs={d: sections.get(d,{}).get("I_B",0.0)/MW for d in SEC_DISP}
        return total,clk,spec,eps,secs

    print("\n>> PHASE 2: validating in REAL Aspen (this opens the flowsheet)...")
    asp=open_aspen()
    try:
        # SINGLE base, write-then-solve (the pipeline's one branch, ~110.5 MW).
        # The optimum is evaluated by the IDENTICAL Reinit->write->solve path, so the
        # comparison is on one consistent branch (no untouched/consistent split).
        print("   - base case: write-then-solve at base inputs (single branch, ~110.5 MW)...")
        base_inputs={c:base_x[i] for i,c in enumerate(INPUT_COLS)}
        bw_sec,bw_pl=evaluate_case(asp, inputs=base_inputs)
        bt,bc,bs,be,bsecs=summarise(bw_sec,bw_pl)

        print("   - PSO optimum: write-then-solve at optimised inputs...")
        opt_inputs={c:gbest[i] for i,c in enumerate(INPUT_COLS)}
        o_sec,o_pl=evaluate_case(asp, inputs=opt_inputs)
        ot,oc,os_,oe,osecs=summarise(o_sec,o_pl)
    finally:
        try: asp.Close(); asp.Quit()
        except Exception: pass

    def pct(a,b): return 100.0*(b-a)/a if a else float("nan")

    # Single-branch comparison: optimum vs the write-then-solve base. Both use the
    # identical Reinit->write->solve path, so there is ONE base value everywhere
    # (matches Table 3 / calculator / dataset), and the change is the true
    # optimisation effect.

    # CSV (Table 6 source)
    import csv
    with open(os.path.join(PSO_DIR,"validate_optimum.csv"),"w",newline="") as fh:
        wcsv=csv.writer(fh)
        wcsv.writerow(["quantity","base_aspen","optimum_aspen","change_pct"])
        wcsv.writerow(["total_irreversibility_MW", bt, ot, pct(bt,ot)])
        wcsv.writerow(["clinker_exergy_MW", bc, oc, pct(bc,oc)])
        wcsv.writerow(["specific_irreversibility", bs, os_, pct(bs,os_)])
        wcsv.writerow(["overall_efficiency_pct", be, oe, pct(be,oe)])
        for d in SEC_DISP:
            wcsv.writerow(["I_B_%s_MW"%d.replace(" ",""), bsecs[d], osecs[d], pct(bsecs[d],osecs[d])])

    # ---- validated-result figures (base vs optimum), from the SAME numbers ----
    base_secs=[bsecs[d] for d in SEC_DISP]
    opt_secs =[osecs[d] for d in SEC_DISP]
    plot_section_bars(SEC_NICE, base_secs, opt_secs)
    plot_waterfall(SEC_NICE, base_secs, opt_secs, bt, ot)
    print(">> wrote pso_section_bars and pso_waterfall (.png/.pdf)")

    # readable report + sanity flags
    with open(os.path.join(PSO_DIR,"validate_optimum.txt"),"w") as fh:
        def w(m): print(m); fh.write(m+"\n")
        w("="*78); w("STEP 4 PHASE 2 - ASPEN-VALIDATED optimum (REAL model)"); w("="*78)
        if FIX_THROUGHPUT:
            w("mode: FIX_THROUGHPUT (constant duty) - pinned %s, clinker +-%.0f%%"
              %(", ".join(FIXED_INPUTS), CLK_BAND*100))
        w("comparison: optimum vs the write-then-solve base (single branch, ~110.5 MW),")
        w("            the same base used in Table 3 / calculator / dataset.")
        w("")
        w("%-24s %12s %12s %9s"%("quantity","base","optimum","change"))
        w("-"*62)
        w("%-24s %12.3f %12.3f %8.1f%%"%("total irreversibility MW", bt, ot, pct(bt,ot)))
        w("%-24s %12.3f %12.3f %8.1f%%"%("clinker exergy MW",        bc, oc, pct(bc,oc)))
        w("%-24s %12.4f %12.4f %8.1f%%"%("specific irrev MW/MW",     bs, os_, pct(bs,os_)))
        w("%-24s %12.1f %12.1f"        %("overall efficiency %",     be, oe))
        w("-"*62)
        w("per-section irreversibility (MW): base -> optimum")
        for d in SEC_DISP:
            w("  %-14s %9.3f -> %9.3f  (%+5.1f%%)"%(d,bsecs[d],osecs[d],pct(bsecs[d],osecs[d])))
        w("")
        # sanity flags (single-branch comparison)
        clk_chg=pct(bc,oc)
        if abs(clk_chg) <= 1.5:
            w("[OK] clinker change = %+.1f%% (production effectively held at constant duty)."%clk_chg)
        else:
            w("[CAUTION] clinker moves %+.1f%% - a reaction-extent variable may still be"%clk_chg)
            w("          free; review before reporting.")
        if be>0 and oe/max(be,1e-9) > 1.5:
            w("[CAUTION] optimum efficiency >1.5x base - investigate.")
        else:
            w("[OK] overall efficiency stays near base (%.1f%% -> %.1f%%)."%(be,oe))
        if flagged:
            w("[NOTE] free inputs pinned at a box bound: %s"
              %", ".join("%s(%s)"%(c,wd) for c,wd in flagged))
            w("       many edge-pinned inputs => the +-%.0f%% box itself is binding;"%(ENVELOPE*100))
            w("       consider tightening the box if a smoother interior optimum is wanted.")
        w("")
        w(">> TABLE 6: report total-I and specific-I reduction at CONSTANT DUTY, using")
        w("   the change column (optimum vs base). Single base = %.1f MW (write-then-solve)."%bt)
    print(">> PHASE 2 done. Validated artifacts in:", PSO_DIR)

# ============================================================ MAIN =============
def main():
    base_x, gbest, flagged = phase1_pso()
    if RUN_VALIDATION:
        phase2_validate(base_x, gbest, flagged)
    else:
        print("\n>> validation disabled (RUN_VALIDATION=False); Phase-1 results only.")
    print("\nDONE.")

if __name__=="__main__":
    main()
