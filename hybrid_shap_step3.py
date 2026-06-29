#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STEP 3 (HYBRID)  -  SHAP sensitivity analysis on the per-section hybrid surrogate
================================================================================
The Step-2 hybrid routes each output to the model that won on validation:
RF for the drying-dominated mills (RawMill, CoalMill) and the clinker/efficiency
plant outputs; ANN for the thermal sections (Preheater, Calciner, Kiln, Cooler)
and specific irreversibility. SHAP therefore explains EACH output with ITS OWN
model:

  * RF-routed outputs  -> shap.TreeExplainer (exact, on RAW inputs)
  * ANN-routed outputs -> shap.GradientExplainer (on SCALED inputs)

Per-output importances are mean(|SHAP|) over the explained samples, NORMALIZED so
each output's 26 indices sum to 1. That per-output normalization makes the ANN
(scaled-input) and RF (raw-input) importances directly comparable as relative
rankings, so the per-section and plant-wide tables are consistent across models.

Produces (in <ASPEN_FOLDER>/ANN_models/HYBRID/shap/):
  shap_importance.csv            : 26 inputs x 18 section metrics (normalized)
  shap_top_inputs.csv            : Step-4 reduced sets (index > THRESHOLD)
  shap_importance_by_section.csv : per-section aggregate (mean over 3 metrics)
  ranked_sensitivity_by_section.csv / top_drivers_summary.csv /
  ranked_sensitivity_table.txt   : paper-ready ranked tables
  shap_specific_irrev.csv / .txt : plant-wide SHAP on specific irreversibility
  plots/ : per-section bar charts (bar_<Section>_I_B.png), the plant-wide
           specific-irreversibility bar (bar_plant_spec_I.png), AND the
           per-section importance heatmap (heatmap_section_importance.png/.pdf)

RUN:  pip install shap tensorflow scikit-learn pandas joblib matplotlib
      python hybrid_shap_step3.py
Requires Step-2 hybrid artifacts in <ASPEN_FOLDER>/ANN_models/HYBRID/:
  hybrid_ann.keras, scaler_X.pkl, scaler_y.pkl, hybrid_bundle.joblib
================================================================================
"""
import os, sys, numpy as np, pandas as pd

# ============================================================ CONFIG ===========
ASPEN_FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
DATASET_CSV  = os.path.join(ASPEN_FOLDER, "ann_dataset.csv")
MODEL_DIR    = os.path.join(ASPEN_FOLDER, "ANN_models", "HYBRID")
SHAP_DIR     = os.path.join(MODEL_DIR, "shap")
PLOT_DIR     = os.path.join(SHAP_DIR, "plots")

SEED         = 42
USE_STATUS   = ("OK",)
THRESHOLD    = 0.05
BACKGROUND_N = 100           # background for GradientExplainer (ANN)
NSAMPLE_EXPL = 200           # samples explained

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
SECTION_COLS = [c for s in SECTIONS for c in sec_outputs(s)]       # 18
PLANT_OUTPUTS = ["clinker_ex_MW","plant_spec_I_MJkg","plant_eps"]  # 3
MODEL_OUTPUTS = SECTION_COLS + PLANT_OUTPUTS                       # 21 (Step-2 order)
OUTPUT_COLS  = SECTION_COLS                                        # 18 analysed per-section
PLANT_SPEC_OUTPUT = "plant_spec_I_MJkg"

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
# readable section names for figure axes
SECTION_LABELS = {
 "RawMill":"Raw mill","CoalMill":"Coal mill","Preheater":"Preheater",
 "Calciner":"Calciner","Kiln":"Kiln","ClinkerCooler":"Clinker cooler",
}
LBL=[INPUT_LABELS[c] for c in INPUT_COLS]

# ============================================================ SETUP ============
np.random.seed(SEED)
os.makedirs(SHAP_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import load_model
import joblib, shap
import matplotlib; matplotlib.use("Agg")
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
def load_hybrid():
    need=["hybrid_ann.keras","scaler_X.pkl","scaler_y.pkl","hybrid_bundle.joblib"]
    for fp in need:
        if not os.path.exists(os.path.join(MODEL_DIR,fp)):
            sys.exit("missing Step-2 hybrid artifact: %s (run hybrid_train_step2.py first)"%fp)
    if not os.path.exists(DATASET_CSV): sys.exit("dataset not found: %s"%DATASET_CSV)
    ann=load_model(os.path.join(MODEL_DIR,"hybrid_ann.keras"))
    sx=joblib.load(os.path.join(MODEL_DIR,"scaler_X.pkl"))
    sy=joblib.load(os.path.join(MODEL_DIR,"scaler_y.pkl"))
    d=joblib.load(os.path.join(MODEL_DIR,"hybrid_bundle.joblib"))
    df=pd.read_csv(DATASET_CSV)
    if "status" in df.columns: df=df[df["status"].isin(USE_STATUS)].reset_index(drop=True)
    X=df[INPUT_COLS].astype(float).values
    return ann,sx,sy,d,X

# ============================================================ TABLES ===========
def write_sensitivity_tables(imp_df, sec_imp, topk=5):
    long_rows=[]
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False)
        for rank,(inp,val) in enumerate(v.items(),1):
            long_rows.append(dict(section=s,rank=rank,input=INPUT_LABELS[inp],
                                  input_key=inp,sensitivity_index=round(float(val),4)))
    pd.DataFrame(long_rows).to_csv(os.path.join(SHAP_DIR,"ranked_sensitivity_by_section.csv"),index=False)
    summ=[]
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False); entry={"section":s}
        for r in range(topk):
            inp=v.index[r]; entry["rank%d"%(r+1)]="%s (%.3f)"%(INPUT_LABELS[inp],v[inp])
        summ.append(entry)
    pd.DataFrame(summ).to_csv(os.path.join(SHAP_DIR,"top_drivers_summary.csv"),index=False)
    lines=["RANKED SENSITIVITY - top %d input drivers per section"%topk,
           "(normalized SHAP importance, mean over the section's 3 exergy metrics; hybrid surrogate)",
           "="*78]
    hdr="%-15s | "%"Section"+" | ".join("Rank %d"%(i+1) for i in range(topk))
    lines+= [hdr,"-"*len(hdr)]
    for s in SECTIONS:
        v=sec_imp[s].sort_values(ascending=False)
        cells=["%s %.3f"%(INPUT_LABELS[v.index[r]],v[v.index[r]]) for r in range(topk)]
        lines.append("%-15s | "%s+" | ".join(cells))
    lines.append("="*78)
    overall=imp_df.mean(axis=1).sort_values(ascending=False)
    lines.append("\nOVERALL plant-wide input ranking (mean over all 18 section outputs):")
    for rank,(inp,val) in enumerate(overall.items(),1):
        lines.append("  %2d. %-16s %.4f"%(rank,INPUT_LABELS[inp],val))
    txt="\n".join(lines)
    with open(os.path.join(SHAP_DIR,"ranked_sensitivity_table.txt"),"w",encoding="utf-8") as fh:
        fh.write(txt+"\n")
    print("\n"+txt)

# ============================================================ FIGURES ==========
def section_heatmap(sec_imp, fname="heatmap_section_importance.png"):
    """Per-section SHAP importance heatmap: 26 inputs x 6 sections (3-metric mean).
    Built from the same aggregate as the ranked tables, so figure and tables agree.
    Rows ordered by overall influence; cells at/above THRESHOLD are annotated."""
    M=sec_imp[SECTIONS].values.astype(float)                 # 26 x 6
    keys=list(sec_imp.index)
    order=np.argsort(-M.mean(axis=1))                        # rows by overall influence
    M=M[order]; ylabels=[INPUT_LABELS[keys[i]] for i in order]
    xlabels=[SECTION_LABELS[s] for s in SECTIONS]

    fig,ax=plt.subplots(figsize=(7.0,9.6))
    vmax=float(M.max()) if M.size else 1.0
    im=ax.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=vmax)

    ax.set_xticks(range(len(SECTIONS))); ax.set_xticklabels(xlabels,rotation=30,ha="right")
    ax.set_yticks(range(len(ylabels)));  ax.set_yticklabels(ylabels)
    # thin white gridlines between cells
    ax.set_xticks(np.arange(-.5,len(SECTIONS),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(ylabels),1),minor=True)
    ax.grid(which="minor",color="white",lw=0.8); ax.tick_params(which="minor",length=0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]>=THRESHOLD:
                ax.text(j,i,"%.2f"%M[i,j],ha="center",va="center",
                        color="white" if M[i,j]>0.5*vmax else "black")

    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03)
    cb.set_label("Normalised SHAP importance"); cb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR,fname),bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_DIR,fname.replace(".png",".pdf")),bbox_inches="tight")
    plt.close(fig)

# ============================================================ SHAP =============
def ann_shap_all(ann, bg_s, Xexpl_s):
    """GradientExplainer on the ANN (scaled space) -> list of 21 arrays (n,26)."""
    expl=shap.GradientExplainer(ann, bg_s)
    sv=expl.shap_values(Xexpl_s)
    if isinstance(sv,list): sv_all=sv
    else: sv_all=[sv[:,:,k] for k in range(sv.shape[-1])]
    if len(sv_all)!=len(MODEL_OUTPUTS):
        sys.exit("ANN SHAP returned %d outputs, expected %d"%(len(sv_all),len(MODEL_OUTPUTS)))
    return sv_all

def rf_shap(forest, Xexpl_raw):
    """TreeExplainer on one forest (raw space) -> (n,26)."""
    te=shap.TreeExplainer(forest)
    sv=te.shap_values(Xexpl_raw, check_additivity=False)
    return np.asarray(sv)

def main():
    ann,sx,sy,d,X = load_hybrid()
    forests=d["forests"]; routing=d["routing"]; out_cols=d["output_cols"]
    # routing/forests are aligned to out_cols == MODEL_OUTPUTS
    route={c:routing[i] for i,c in enumerate(out_cols)}

    n=len(X); rng=np.random.default_rng(SEED)
    bg_idx=rng.choice(n,size=min(BACKGROUND_N,n),replace=False)
    ex_idx=rng.choice(n,size=min(NSAMPLE_EXPL,n),replace=False)
    Xexpl_raw=X[ex_idx]; Xbg_raw=X[bg_idx]
    Xs=sx.transform(X); Xexpl_s=Xs[ex_idx]; Xbg_s=Xs[bg_idx]

    print(">> ANN GradientExplainer (scaled) for ANN-routed outputs...")
    ann_sv=ann_shap_all(ann, Xbg_s, Xexpl_s)        # 21 arrays (n,26), scaled-input attributions

    print(">> RF TreeExplainer (raw) for RF-routed outputs...")
    rf_sv={}                                        # output_col -> (n,26)
    for j,c in enumerate(out_cols):
        if route[c]=="rf":
            rf_sv[c]=rf_shap(forests[j], Xexpl_raw)

    # assemble per-output SHAP from the routed model
    def sv_for(c):
        j=out_cols.index(c)
        return ann_sv[j] if route[c]=="ann" else rf_sv[c]

    # ---- per-output normalized importance over the 18 section metrics ----
    imp=np.zeros((len(INPUT_COLS),len(OUTPUT_COLS)))
    for k,c in enumerate(OUTPUT_COLS):
        m=np.mean(np.abs(sv_for(c)),axis=0); s=m.sum()
        imp[:,k]= m/s if s>0 else m
    imp_df=pd.DataFrame(imp,index=INPUT_COLS,columns=OUTPUT_COLS)
    imp_df.to_csv(os.path.join(SHAP_DIR,"shap_importance.csv"))
    print(">> wrote shap_importance.csv (26 x 18, normalized, per-output routed model)")

    # Step-4 reduced sets
    rows=[]
    for k,oc in enumerate(OUTPUT_COLS):
        keep=[INPUT_COLS[i] for i in range(len(INPUT_COLS)) if imp[i,k]>THRESHOLD]
        rows.append({"output":oc,"n_kept":len(keep),"kept_inputs":";".join(keep)})
    pd.DataFrame(rows).to_csv(os.path.join(SHAP_DIR,"shap_top_inputs.csv"),index=False)

    # per-section aggregate
    sec_imp=pd.DataFrame(index=INPUT_COLS)
    for s in SECTIONS:
        sec_imp[s]=imp_df[sec_outputs(s)].mean(axis=1)
    sec_imp.to_csv(os.path.join(SHAP_DIR,"shap_importance_by_section.csv"))

    # console: top-3 per section's I_B, noting the model used
    print("\nTop-3 input drivers of each section's irreversibility (model in [..]):")
    for s in SECTIONS:
        c="%s_I_B_MW"%s; v=imp_df[c].sort_values(ascending=False)
        top3=", ".join("%s(%.2f)"%(INPUT_LABELS[i],v[i]) for i in v.index[:3])
        print("  %-14s [%s] %s"%(s,route[c].upper(),top3))

    write_sensitivity_tables(imp_df, sec_imp)

    # ---- plant-wide SHAP on specific irreversibility (its routed model) ----
    c=PLANT_SPEC_OUTPUT; sv=sv_for(c)
    m=np.mean(np.abs(sv),axis=0); s=m.sum(); idx=(m/s) if s>0 else m
    impS=pd.Series(idx,index=INPUT_COLS).sort_values(ascending=False)
    pd.DataFrame([dict(rank=r+1,input=INPUT_LABELS[k],input_key=k,
                       sensitivity_index=round(float(impS[k]),4))
                  for r,k in enumerate(impS.index)]
                 ).to_csv(os.path.join(SHAP_DIR,"shap_specific_irrev.csv"),index=False)
    lines=["PLANT-WIDE SHAP - SPECIFIC IRREVERSIBILITY (plant_spec_I_MJkg, MJ/kg) [model: %s]"%route[c].upper(),
           "(complementary to per-section analysis; normalized mean|SHAP|, 26 inputs)","="*74]
    for r,k in enumerate(impS.index,1): lines.append("  %2d. %-16s %.4f"%(r,INPUT_LABELS[k],impS[k]))
    lines.append("="*74)
    open(os.path.join(SHAP_DIR,"shap_specific_irrev.txt"),"w",encoding="utf-8").write("\n".join(lines)+"\n")
    print("\n"+"\n".join(lines))

    # ================================ PLOTS ================================
    # (a) ranked bars for the 6 section I_B + the plant-wide spec-I bar
    def bar(series, title, fname, color="#2e6f95"):
        order=series.sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(7,6))
        ax.barh([INPUT_LABELS[k] for k in order.index][::-1], order.values[::-1], color=color)
        ax.axvline(THRESHOLD,color="grey",ls=":",lw=1)
        ax.set_xlabel("Normalized SHAP importance"); ax.set_title(title)
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR,fname),bbox_inches="tight"); plt.close(fig)
    for s in SECTIONS:
        c2="%s_I_B_MW"%s
        bar(imp_df[c2], "SHAP - %s irreversibility [%s]"%(s,route[c2].upper()), "bar_%s_I_B.png"%s)
    bar(impS, "Plant-wide drivers of specific irreversibility [%s]"%route[PLANT_SPEC_OUTPUT].upper(),
        "bar_plant_spec_I.png", color="#b5651d")

    # (b) per-section importance heatmap (26 inputs x 6 sections, 3-metric mean)
    section_heatmap(sec_imp)
    print(">> wrote heatmap_section_importance.png/.pdf")

    print("\nDONE. Hybrid SHAP artifacts in:", SHAP_DIR)

if __name__=="__main__":
    main()
