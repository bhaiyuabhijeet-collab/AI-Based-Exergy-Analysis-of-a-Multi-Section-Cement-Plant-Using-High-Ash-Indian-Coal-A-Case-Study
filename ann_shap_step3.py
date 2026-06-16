#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 3  -  SHAP sensitivity analysis
================================================================================
Loads the trained single 18-output ANN from Step 2 and computes SHAP
(SHapley Additive exPlanations) values over the 26 process inputs for each of
the 18 exergy outputs. Produces, for every output:

  * a ranked global sensitivity index  = mean(|SHAP|) over the dataset,
    normalized so the 26 indices sum to 1 (so "importance > 0.05" in Step 4
    is a clear, scale-free threshold),
  * a SHAP summary (beeswarm) plot and a ranked bar plot (Times New Roman 12).

It also writes:
  * shap_importance.csv      : 26 inputs x 18 outputs, normalized indices
  * shap_top_inputs.csv      : for each output, inputs with index > THRESHOLD
                               (this is exactly the Step-4 reduced input set)
  * a per-section aggregated importance table (mean over that section's
    3 metrics) for the paper's ranked-sensitivity discussion.

RUN:  pip install shap tensorflow scikit-learn pandas joblib matplotlib
      python ann_shap_step3.py
Requires Step 2 artifacts in  <ASPEN_FOLDER>/ANN_models/ :
  single_18output.keras, scaler_X.pkl, scaler_y.pkl
and the dataset CSV (for the background / explanation samples).
================================================================================
"""
import os, sys, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models")
SHAP_DIR     = os.path.join(MODEL_DIR, "shap")
PLOT_DIR     = os.path.join(SHAP_DIR, "plots")

SEED         = 42
USE_STATUS   = ("OK",)
THRESHOLD    = 0.05          # Step-4 keeps inputs with normalized index > this
BACKGROUND_N = 100           # background samples for the SHAP explainer
NSAMPLE_EXPL = 200           # samples explained (<= dataset size); more = smoother

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
OUTPUT_COLS = [c for s in SECTIONS for c in sec_outputs(s)]   # 18, section-blocked

# short, readable input labels for plots
INPUT_LABELS = {
 "rawmeal_feed[kg/s]":"Raw-meal feed","rmgas_flow[kg/s]":"RM gas flow",
 "rmgas_temp[K]":"RM gas T","coal_feed[kg/s]":"Coal feed",
 "cmgas_flow[kg/s]":"CM gas flow","cmgas_temp[K]":"CM gas T",
 "clrcair_flow[kg/s]":"Cooler air 1","clrcair2_flow[kg/s]":"Cooler air 2",
 "rmfan_T[K]":"RM fan T","rmdry_T[K]":"RM dryer T","gasht_T[K]":"Gas-heater T",
 "cmdry_T[K]":"CM dryer T",
 "phk1_T[K]":"PH-K1 T","phk2_T[K]":"PH-K2 T","phk3_T[K]":"PH-K3 T",
 "phk4_T[K]":"PH-K4 T","phk5_T[K]":"PH-K5 T",
 "phc1_T[K]":"PH-C1 T","phc2_T[K]":"PH-C2 T","phc3_T[K]":"PH-C3 T",
 "phc4_T[K]":"PH-C4 T","phc5_T[K]":"PH-C5 T",
 "clcalc_T[K]":"Calciner T","coal_split[-]":"Coal split",
 "secair_split[-]":"Sec-air split","meal_split[-]":"Meal split",
}
LBL=[INPUT_LABELS[c] for c in INPUT_COLS]

# ============================================================ SETUP ============
np.random.seed(SEED)
os.makedirs(SHAP_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import load_model
import joblib, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

def setup_fonts():
    for fam in ["Times New Roman","Times","DejaVu Serif"]:
        if any(f.name==fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]=fam; break
    else:
        plt.rcParams["font.family"]="serif"
    plt.rcParams.update({
        "font.size":12,"axes.titlesize":12,"axes.labelsize":12,
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

# ============================================================ SHAP =============
def write_sensitivity_tables(imp_df, sec_imp, topk=5):
    """Produce paper-ready ranked-sensitivity tables:
       (1) ranked_sensitivity_by_section.csv  - per section (mean over its 3
           metrics), all 26 inputs ranked, with rank and normalized index.
       (2) top_drivers_summary.csv            - compact: top-k drivers per
           section in one row each (for the main-text table).
       (3) ranked_sensitivity_table.txt       - a formatted, copy-paste table.
    """
    # (1) full ranked per-section
    long_rows=[]
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False)
        for rank,(inp,val) in enumerate(v.items(),1):
            long_rows.append(dict(section=s,rank=rank,
                                  input=INPUT_LABELS[inp],
                                  input_key=inp,
                                  sensitivity_index=round(float(val),4)))
    pd.DataFrame(long_rows).to_csv(
        os.path.join(SHAP_DIR,"ranked_sensitivity_by_section.csv"),index=False)

    # (2) compact top-k summary, one row per section
    summ=[]
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False)
        entry={"section":s}
        for r in range(topk):
            inp=v.index[r]
            entry["rank%d"%(r+1)]="%s (%.3f)"%(INPUT_LABELS[inp],v[inp])
        summ.append(entry)
    pd.DataFrame(summ).to_csv(
        os.path.join(SHAP_DIR,"top_drivers_summary.csv"),index=False)

    # (3) formatted text table for direct paste
    lines=[]
    lines.append("RANKED SENSITIVITY — top %d input drivers per section"%topk)
    lines.append("(normalized SHAP importance, mean over the section's 3 exergy metrics)")
    lines.append("="*78)
    hdr="%-15s | "%"Section" + " | ".join("Rank %d"%(i+1) for i in range(topk))
    lines.append(hdr); lines.append("-"*len(hdr))
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False)
        cells=["%s %.3f"%(INPUT_LABELS[v.index[r]],v[v.index[r]]) for r in range(topk)]
        lines.append("%-15s | "%s + " | ".join(cells))
    lines.append("="*78)
    # also an overall (plant-wide) ranking = mean importance across all 18 outputs
    overall=imp_df.mean(axis=1).sort_values(ascending=False)
    lines.append("\nOVERALL plant-wide input ranking (mean over all 18 outputs):")
    for rank,(inp,val) in enumerate(overall.items(),1):
        lines.append("  %2d. %-16s %.4f"%(rank,INPUT_LABELS[inp],val))
    txt="\n".join(lines)
    with open(os.path.join(SHAP_DIR,"ranked_sensitivity_table.txt"),"w") as fh:
        fh.write(txt+"\n")
    print("\n"+txt)
    print("\n>> tables: ranked_sensitivity_by_section.csv, top_drivers_summary.csv,")
    print("           ranked_sensitivity_table.txt")

def main():
    model,sx,sy,X = load_all()
    Xs = sx.transform(X)                       # model works in scaled input space
    n = len(Xs)
    rng=np.random.default_rng(SEED)
    bg_idx = rng.choice(n, size=min(BACKGROUND_N,n), replace=False)
    ex_idx = rng.choice(n, size=min(NSAMPLE_EXPL,n), replace=False)
    background = Xs[bg_idx]
    Xexpl = Xs[ex_idx]

    print(">> building SHAP explainer (GradientExplainer on the Keras model)...")
    # GradientExplainer is well-suited to Keras MLPs and handles multi-output.
    explainer = shap.GradientExplainer(model, background)
    print(">> computing SHAP values for %d samples x 18 outputs..."%len(Xexpl))
    shap_vals = explainer.shap_values(Xexpl)   # list(18) of (Nexpl,26) OR (Nexpl,26,18)

    # Normalize the return shape to a list of 18 arrays, each (Nexpl, 26)
    if isinstance(shap_vals, list):
        sv_list = shap_vals
    else:
        # array form (Nexpl, 26, 18) -> split on last axis
        sv_list = [shap_vals[:,:,k] for k in range(shap_vals.shape[-1])]

    # ---- global importance = mean(|SHAP|) per input, per output, normalized ----
    imp = np.zeros((len(INPUT_COLS), len(OUTPUT_COLS)))
    for k in range(len(OUTPUT_COLS)):
        m = np.mean(np.abs(sv_list[k]), axis=0)        # (26,)
        s = m.sum()
        imp[:,k] = m/s if s>0 else m
    imp_df = pd.DataFrame(imp, index=INPUT_COLS, columns=OUTPUT_COLS)
    imp_df.to_csv(os.path.join(SHAP_DIR,"shap_importance.csv"))
    print(">> wrote shap_importance.csv (26 inputs x 18 outputs, normalized)")

    # ---- Step-4 reduced sets: inputs with index > THRESHOLD, per output ----
    rows=[]
    for k,oc in enumerate(OUTPUT_COLS):
        keep=[INPUT_COLS[i] for i in range(len(INPUT_COLS)) if imp[i,k]>THRESHOLD]
        rows.append({"output":oc,"n_kept":len(keep),"kept_inputs":";".join(keep)})
    pd.DataFrame(rows).to_csv(os.path.join(SHAP_DIR,"shap_top_inputs.csv"),index=False)
    print(">> wrote shap_top_inputs.csv (Step-4 reduced input sets, threshold %.2f)"%THRESHOLD)

    # ---- per-section aggregated importance (mean over its 3 metrics) ----
    sec_imp=pd.DataFrame(index=INPUT_COLS)
    for s in SECTIONS:
        cols=sec_outputs(s)
        sec_imp[s]=imp_df[cols].mean(axis=1)
    sec_imp.to_csv(os.path.join(SHAP_DIR,"shap_importance_by_section.csv"))

    # ---- plots ----
    # (a) per-output ranked bar of normalized importance
    for k,oc in enumerate(OUTPUT_COLS):
        order=np.argsort(imp[:,k])[::-1]
        fig,ax=plt.subplots(figsize=(7,5))
        ax.barh([LBL[i] for i in order][::-1], imp[order,k][::-1],
                color="#2e6f95")
        ax.axvline(THRESHOLD,color="grey",ls=":",lw=1)
        ax.set_xlabel("Normalized SHAP importance")
        ax.set_title("SHAP importance — %s"%oc.replace("_"," "))
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR,"bar_%s.png"%oc),bbox_inches="tight")
        plt.close(fig)

    # (b) SHAP beeswarm summary for the 6 irreversibility outputs (the headline ones)
    for s in SECTIONS:
        k=OUTPUT_COLS.index("%s_I_B_MW"%s)
        plt.figure()
        shap.summary_plot(sv_list[k], Xexpl, feature_names=LBL, show=False,
                          plot_size=(7,5))
        plt.title("SHAP summary — %s irreversibility"%s)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,"beeswarm_%s_I_B.png"%s),
                    bbox_inches="tight",dpi=300)
        plt.close()

    # (c) per-section aggregated importance heatmap-style bar (top drivers)
    fig,ax=plt.subplots(figsize=(8,6))
    # show top-10 inputs by overall mean importance
    overall=imp.mean(axis=1); top=np.argsort(overall)[::-1][:10]
    width=0.8/len(SECTIONS)
    xpos=np.arange(len(top))
    for j,s in enumerate(SECTIONS):
        ax.bar(xpos+j*width, sec_imp[s].values[top], width, label=s)
    ax.set_xticks(xpos+0.4-width/2); ax.set_xticklabels([LBL[i] for i in top],rotation=90)
    ax.set_ylabel("Normalized SHAP importance"); ax.legend(ncol=2,fontsize=9)
    ax.set_title("Top input drivers of exergy destruction, by section")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR,"section_drivers.png"),bbox_inches="tight")
    plt.close(fig)

    print(">> plots (Times New Roman 12) ->", PLOT_DIR)
    # quick console summary: top-3 drivers of each section's irreversibility
    print("\nTop-3 input drivers of each section's irreversibility (İ):")
    for s in SECTIONS:
        col="%s_I_B_MW"%s; v=imp_df[col].sort_values(ascending=False)
        top3=", ".join("%s(%.2f)"%(INPUT_LABELS[i],v[i]) for i in v.index[:3])
        print("  %-14s %s"%(s,top3))

    # ============================ RANKED-SENSITIVITY TABLES ==================
    write_sensitivity_tables(imp_df, sec_imp)

    print("\nDONE. SHAP artifacts in:", SHAP_DIR)

if __name__=="__main__":
    main()
