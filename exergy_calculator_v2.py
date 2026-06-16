#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXERGY CALCULATOR  v2.2 -  Multi-Section Cement Plant, High-Ash Indian Coal
================================================================================
WHAT CHANGED vs the previous (final) script
--------------------------------------------------------------------------------
1) PHYSICAL EXERGY (MIXED + CISOLID) now uses the DEAD-STATE REFERENCE-STREAM
   method (Hinderink et al., 1996, Chem. Eng. Sci. 51:4693). For each distinct
   stream-composition family a duplicate stream is created in the Aspen model
   (Dupl block) and flashed to T0=298.15 K, P0=101.325 kPa in a Heater. Its
   HMX_MASS / SMX_MASS per substream ARE h0 and s0 - same property method
   (SOLIDS), same reference basis, so formation terms cancel exactly, pressure
   and dead-state phase change (moisture condensation) are included exactly,
   and physical exergy is consistent with the simulated entropy balance.
   -> No property sets needed. SMX_MASS at a stream's OWN state always reads
      (your export proves it); only the prop-set "entropy at another T" failed.
   -> Because Dupl tracks composition automatically, the method holds unchanged
      across all 500 sampled runs.

2) NC COAL: Aspen has no entropy model for nonconventional solids (this is a
   hard product limitation, same reason ExerPy ignores NC). Physical exergy
   stays closed-form (m*Cp*[(T-T0)-T0 ln(T/T0)], Cp=1134 from two-point check,
   <0.05% of chemical). The literature absolute entropy (Eisermann et al. 1980)
   is used ONLY in the Gouy-Stodola cross-check, never in the balance route.

3) COAL CHEMICAL EXERGY: corrected Szargut-Styrylska. The previous script mixed
   the linear numerator (valid O/C<=0.667) with the biomass denominator
   (1-0.3035 O/C, valid 0.667<O/C<=2.67), inflating beta by ~6%. For this coal
   O/C=0.184 -> beta=1.0697 (was 1.1331); ex=21.54 MJ/kg at NCV=20.1 MJ/kg.

4) CLINKER MINERAL CHEMICAL EXERGIES: recomputed from traceable data
   (Hanein, Glasser & Bannerman, Cem. Concr. Res. 132 (2020) 106043):
   C3S 211.5, C2S 93.5, C3A 512.1, C4AF 582.6 kJ/mol (see Annexure A2).

5) MIXING TERM RT0*sum(x ln x) applied to MIXED (gas) substream only; CISOLID
   treated as mechanical mixture of pure crystalline phases (Szargut).

6) SECTION RESULTS: irreversibility from TWO INDEPENDENT ROUTES with residual:
     Route B (primary): I = sum Ex_in + W_in + sum Q_b(1 - T0/T_b) - sum Ex_out
     Route A (check):   I = T0 * S_gen   (Aspen entropies + coal-entropy bridge)
   Efficiency is now the KOTAS FUNCTIONAL efficiency eps = Ex_product/Ex_fuel
   with the product/fuel definitions from the paper (per section); the gross
   ratio Ex_out/Ex_in is reported alongside for reference. IP = (1-eps)*I.
   The abs(I) guard and the 100% efficiency cap are REMOVED: negative values
   are now flagged loudly instead of hidden, because they indicate data errors.

7) BONUS VALIDATION: if Aspen's native exergy is enabled (Setup > Calculation
   Options > "Perform exergy calculations"... sets STRM_UPP\\EXERGYMS), the
   script reads it per substream where available and writes it next to the
   computed values (this is exactly the node ExerPy reads for MIXED streams).

--------------------------------------------------------------------------------
ONE-TIME ASPEN MODEL EDIT (do once; survives all 500 runs automatically)
--------------------------------------------------------------------------------
Insert a Dupl block on each stream listed below; route the duplicate into a
Heater specified at T = 25 C, P = 1.01325 bar (Valid phases: Vapor-Liquid for
gas refs; default for solids). Name the Heater outlet stream exactly as the
"DS stream" name. The main flowsheet is unaffected (Dupl is pass-through).

   parent stream    Dupl name   Heater name   DS stream (read by this script)
   RAWMEAL          DUP-RMEA    HT0-RMEA      DS-RMEAL   (raw-meal solid family)
   CLSOL            DUP-CLSO    HT0-CLSO      DS-CLSOL   (calcined meal)
   CLINKOUT         DUP-CLNK    HT0-CLNK      DS-CLNK    (clinker family)
   PHK1GAS          DUP-KGAS    HT0-KGAS      DS-KGAS    (kiln-string gas family)
   PHC1GAS          DUP-CGAS    HT0-CGAS      DS-CGAS    (calciner-string gas family)
   RMINLETG         DUP-RMG     HT0-RMG       DS-RMG
   RMEXHUST         DUP-RMX     HT0-RMX       DS-RMX
   CMINLETG         DUP-CMG     HT0-CMG       DS-CMG
   CMEXHUST         DUP-CMX     HT0-CMX       DS-CMX
   RM-INLET         DUP-RMIN    HT0-RMIN      DS-RMIN    (handles CISOLID+MIXED together)

Streams already at the dead state need no clone (they are their own reference):
   CLRCAIR, CLRCAIR2 (ambient air, also reference for SECAIRNW/TAIRNW), CM-INLET.
Until the clones exist, the script FALLS BACK to the previous Cp-based method
per stream and tags the row "FALLBACK" so nothing breaks meanwhile.

RUN:  pip install pywin32 openpyxl
      python exergy_calculator_v2.py
================================================================================
"""
import os, sys, math, traceback

# ============================================================== CONFIG ========
FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
BKP      = os.path.join(FOLDER, "new cmill rmill preheater calciner kiln cooler.bkp")
OUT_XLSX = os.path.join(FOLDER, "exergy_results_v2.xlsx")

T0 = 298.15            # K
P0 = 101325.0          # Pa
R  = 8.314             # J/mol-K
T_SHELL = 635.0        # K  kiln shell (literature default)

COMPONENTS = ["CARBO-01","OXYGE-01","WATER","NITRO-01","CARBO-02","NITRI-01",
              "SULFU-01","CALCI-01","CALCI-02","SILIC-01","ALUMI-01","HEMAT-01",
              "MAGNE-01","MAGNE-02","TRICA-01","OLIVI-01","(CAO)-01","C4AF",
              "COAL-ASH","C","H2"]
NC_COMP = "COAL-IND"
SUBS = ["MIXED","CISOLID","NC"]

MW = {"CARBO-01":44.01,"OXYGE-01":32.00,"WATER":18.015,"NITRO-01":28.014,
      "CARBO-02":28.01,"NITRI-01":30.006,"SULFU-01":64.066,"CALCI-01":100.087,
      "CALCI-02":56.077,"SILIC-01":60.083,"ALUMI-01":101.961,"HEMAT-01":159.688,
      "MAGNE-01":84.314,"MAGNE-02":40.304,"TRICA-01":228.317,"OLIVI-01":172.240,
      "(CAO)-01":270.193,"C4AF":485.965,"COAL-ASH":60.08,"C":12.011,"H2":2.016}

# =================================================== CHEMICAL EXERGY DATA =====
# Standard chemical exergy ex0_ch [kJ/mol], Szargut Reference Environment II
# (Szargut, Morris & Steward 1988; Kotas 1995 App. A). Clinker minerals from
# oxides + dG_f,ox computed from Hanein, Glasser & Bannerman (2020) dataset:
#   ex0(mineral) = sum(nu*ex0_oxide) + dG_f,ox ,  dG = dH - T0*dS  (Annexure A2)
CHEM_EX = {
    "CARBO-01": 19.87,   "OXYGE-01": 3.97,   "WATER": 9.50,   # H2O vapour
    "NITRO-01": 0.72,    "CARBO-02": 275.10, "NITRI-01": 88.90,
    "SULFU-01": 313.40,  "CALCI-01": 1.00,   "CALCI-02": 110.20,
    "SILIC-01": 1.90,    "ALUMI-01": 200.40, "HEMAT-01": 16.50,
    "MAGNE-01": 37.90,   "MAGNE-02": 66.80,  "COAL-ASH": 0.00,
    "C": 410.26,         "H2": 236.10,
    # clinker phases - Annexure A2 (Hanein et al. 2020 derived)
    "TRICA-01": 211.5,   # C3S : 3*110.2 + 1.9   - 121.0
    "OLIVI-01": 93.5,    # C2S : 2*110.2 + 1.9   - 128.8
    "(CAO)-01": 512.1,   # C3A : 3*110.2 + 200.4 -  18.9
    "C4AF":     582.6,   # C4AF: 4*110.2 + 200.4 + 16.5 - 75.1
}
EX_H2O_LIQ = 0.90        # kJ/mol, liquid water at dead state (Annexure A1)
DG_OX = {"TRICA-01":-121.0, "OLIVI-01":-128.8, "(CAO)-01":-18.9, "C4AF":-75.1}

# ---- Coal (Szargut-Styrylska, corrected piecewise; Annexure A3) ----
NCV_DRY   = 20100.0e3   # J/kg dry. MUST be the NET (lower) CV. If the Boie
                        # 4801 kcal/kg figure is the GROSS CV, set 19330.0e3.
USE_MODEL_NCV = True    # derive the NCV embedded in HCOALGEN from the model's
                        # own coal enthalpy at 298.15 K and use it for Szargut,
                        # guaranteeing energy/exergy alignment (recommended).
_NCV_ACTIVE = [NCV_DRY] # runtime holder (set in main)
_HF_COAL    = [None]    # coal formation enthalpy at 298.15 K, J/kg (from model)
USE_CONSISTENT_COAL_EX = True
# Rigorous Szargut compound-formula coal exergy from the model's own hf:
#   ex = [hf - T0*(s0_abs - sum nu*s_el)] + sum nu*ex_el
# This makes the coal element-consistent with Aspen's formation-entropy
# convention; the beta correlation is printed alongside for comparison.
_COAL_ELS   = [None]    # (sum nu*s_el [J/kg-K], sum nu*ex_el [J/kg]) from ultanal
S_EL  = {"C":5.74,"H2":130.68,"O2":205.152,"N2":191.609,"S":32.054}  # J/mol-K
EX_EL = {"C":410.26e3,"H2":236.10e3,"O2":3.97e3,"N2":0.72e3,"S":609.6e3}  # J/mol
EX_SULFUR = 9417.0e3    # J/kg-S
COAL_CP   = 1134.0      # J/kg-K (two-point CM-INLET 298K / CMDRYCOL 343K)
COAL_S0   = 1050.0      # J/kg-K absolute entropy of solid coal, Eisermann et
                        # al. (1980) estimate; used ONLY in Route-A cross-check.

# ---- Fallback solid Cp [J/kg-K] at 298 K (Annexure A4-1) -------------------
# Used ONLY when a dead-state reference stream is not yet present in the model.
MINERAL_CP = {
    "CALCI-01": 834.0, "CALCI-02": 750.0, "SILIC-01": 742.0, "ALUMI-01": 775.0,
    "HEMAT-01": 650.0, "MAGNE-01": 896.0, "MAGNE-02": 924.0,
    "TRICA-01": 755.0, "OLIVI-01": 745.0, "(CAO)-01": 760.0, "C4AF": 782.0,
    "COAL-ASH": 840.0,
}
DEFAULT_SOLID_CP = 800.0
GAS_CP_SPECIES = {"CARBO-01":846.0,"OXYGE-01":918.0,"WATER":1864.0,
                  "NITRO-01":1040.0,"CARBO-02":1040.0,"NITRI-01":995.0,
                  "SULFU-01":622.0}

# ============================================ DEAD-STATE REFERENCE MAP ========
# stream -> dead-state reference stream whose HMX_MASS/SMX_MASS give h0,s0.
# Streams mapping to themselves are already at (T0,P0).
DS_REF = {
    # raw-meal solid family (composition preserved through preheater SSplits)
    "RAWMEAL":"DS-RMEAL","RMEALK":"DS-RMEAL","RMEALC":"DS-RMEAL",
    "PHK1SOL":"DS-RMEAL","PHC1SOL":"DS-RMEAL","PHK5SOL":"DS-RMEAL","PHC5SOL":"DS-RMEAL",
    # calcined meal & clinker
    "CLSOL":"DS-CLSOL",
    "CLINKER":"DS-CLNK","CLINKERC":"DS-CLNK","CLINKOUT":"DS-CLNK",
    # combustion-gas families (verified identical composition along each string)
    "KILNGAS":"DS-KGAS","PHK1GAS":"DS-KGAS",
    "CLGAS":"DS-CGAS","PHC1GAS":"DS-CGAS",
    # mill gases (composition differs inlet vs exhaust -> separate refs)
    "RMINLETG":"DS-RMG","RMEXHUST":"DS-RMX",
    "CMINLETG":"DS-CMG","CMEXHUST":"DS-CMX",
    # mixed solid+gas feed
    "RM-INLET":"DS-RMIN",
    # air family: ambient cooler air IS the dead state for heated air streams
    "SECAIRNW":"CLRCAIR","TAIRNW":"CLRCAIR","CLRHOT":"CLRCAIR",
    "CLRCAIR":"CLRCAIR","CLRCAIR2":"CLRCAIR2","CLREXH":"CLRCAIR",
    # at dead state already
    "CM-INLET":"CM-INLET",
}
COMP_TOL = 0.02   # warn if stream/reference mass-fraction mismatch exceeds this

# ============================================================ STREAM MAP =======
SECTIONS = {
 "Raw Mill": {"in":["RM-INLET","RMINLETG"], "out":["RAWMEAL","RMEXHUST"],
   "work":["RM-FAN","RM-DRYER"]},
 "Coal Mill": {"in":["CM-INLET","CMINLETG"], "out":["CMDRYCOL","CMEXHUST"],
   "work":["GASHEATE","CM-DRYER"]},
 "Preheater": {"in":["RMEALK","KILNGAS","RMEALC","CLGAS"],
   "out":["PHK5SOL","PHK1GAS","PHC5SOL","PHC1GAS"], "work":[]},
 "Calciner": {"in":["CALCOAL","TAIRNW","PHK5SOL","PHC5SOL"],
   "out":["CLGAS","CLSOL"], "work":[]},
 "Kiln": {"in":["KILNCOAL","SECAIRNW","CLSOL"], "out":["KILNGAS","CLINKERC"],
   "work":[], "heat_loss":[("KLNLOSS",T_SHELL)]},
 "Clinker Cooler": {"in":["CLINKERC","CLRCAIR","CLRCAIR2"],
   "out":["SECAIRNW","TAIRNW","CLINKOUT","CLREXH"], "work":[]},
}
# ---- heat/work blocks per section --------------------------------------------
# kind "work": electrical input modeled as a Heater duty -> exergy = Q (Q>0).
#              If such a block has Q<0 it is treated as heat at its outlet T.
# kind "heat": boundary heat at the block outlet temperature; signed
#              (Q<0 leaving the section, Q>0 entering). Enters BOTH routes:
#              Route B: +Q(1-T0/Tb)   Route A: S_gen -= Q/Tb
# This list covers every duty-carrying block in the flowsheet; with it, each
# section's energy balance closes exactly (verified against the Aspen export):
# the PH-HT cyclone heaters export ~132 MW, the RYield coal-decomposition
# blocks (CLDCMP/KLDECP) absorb ~174 MW, KLNLOSS is the kiln shell loss, and
# CLRFIN is the cooler's final heat rejection.
SECTION_BLOCKS = {
 "Raw Mill":      [("RM-FAN","work"),("RM-DRYER","work")],
 "Coal Mill":     [("GASHEATE","work"),("CM-DRYER","heat")],
 "Preheater":     [("PH-HT-K1","heat"),("PH-HT-K2","heat"),("PH-HT-K3","heat"),
                   ("PH-HT-K4","heat"),("PH-HT-K5","heat"),
                   ("PH-HT-C1","heat"),("PH-HT-C2","heat"),("PH-HT-C3","heat"),
                   ("PH-HT-C4","heat"),("PH-HT-C5","heat")],
 "Calciner":      [("CLCALC","heat"),("CLDCMP","heat"),("CLCOMB","heat")],
 "Kiln":          [("KLDECP","heat"),("KLCALC1","heat"),("KLCALC2","heat"),
                   ("KLMVASH","heat"),("KLCOMB","heat"),("KLNLOSS","heat")],
 "Clinker Cooler":[("CLRFIN","heat")],
}
# fallback outlet temperatures [K] if the block T node cannot be read
FALLBACK_T = {"PH-HT-K1":593.15,"PH-HT-K2":753.15,"PH-HT-K3":903.15,
              "PH-HT-K4":973.15,"PH-HT-K5":1023.15,"PH-HT-C1":593.15,
              "PH-HT-C2":753.15,"PH-HT-C3":903.15,"PH-HT-C4":973.15,
              "PH-HT-C5":1023.15,"CLCALC":1173.15,"CLDCMP":383.15,
              "CLCOMB":3010.0,"KLDECP":383.15,"KLCALC1":1867.0,
              "KLCALC2":1866.0,"KLMVASH":383.15,"KLCOMB":2047.0,
              "KLNLOSS":1728.9,"CLRFIN":398.15,"CM-DRYER":343.15,
              "GASHEATE":438.15,"RM-FAN":438.15,"RM-DRYER":363.15}
ENERGY_CLOSURE_TOL = 0.5e6   # W; flag sections whose energy balance misses this

# Kotas FUNCTIONAL efficiency, with a section-appropriate basis:
#  basis="total"  eps = [sum(p+)-sum(p-)] / [W? + sum(f+)-sum(f-)] on TOTAL exergy
#  basis="ph"     same increments evaluated on PHYSICAL exergy only. Used for the
#                 two drying mills: moisture migrates solid->gas, so total-exergy
#                 increments merely shuffle the water's chemical exergy between
#                 streams and turn negative; the mills' true duty (grinding +
#                 drying + heating) is physical, so the physical basis is the
#                 meaningful rational efficiency for them.
#  Kiln: increment form is NOT meaningful because clinkerization LOWERS chemical
#        exergy (dG_f,ox < 0 for all four phases, Annexure A2) - the clinker is
#        thermodynamically downhill from free lime + oxides. The kiln efficiency
#        is therefore defined as total product over total functional input:
#        eps = Ex(CLINKERC) / [Ex(CLSOL)+Ex(KILNCOAL)+Ex(SECAIRNW)].
EFF_DEF = {
 "Raw Mill":      {"p_plus":["RAWMEAL"],            "p_minus":["RM-INLET"],
                   "f_plus":["RMINLETG"],            "f_minus":["RMEXHUST"],
                   "f_work":True,  "basis":"ph"},
 "Coal Mill":     {"p_plus":["CMDRYCOL"],           "p_minus":["CM-INLET"],
                   "f_plus":["CMINLETG"],            "f_minus":["CMEXHUST"],
                   "f_work":True,  "basis":"ph"},
 "Preheater":     {"p_plus":["PHK5SOL","PHC5SOL"],  "p_minus":["RMEALK","RMEALC"],
                   "f_plus":["KILNGAS","CLGAS"],     "f_minus":["PHK1GAS","PHC1GAS"],
                   "f_work":False, "basis":"total"},
 "Calciner":      {"p_plus":["CLSOL"],              "p_minus":["PHK5SOL","PHC5SOL"],
                   "f_plus":["CALCOAL","TAIRNW"],    "f_minus":[],
                   "f_work":False, "basis":"total"},
 "Kiln":          {"p_plus":["CLINKERC"],           "p_minus":[],
                   "f_plus":["CLSOL","KILNCOAL","SECAIRNW"], "f_minus":[],
                   "f_work":False, "basis":"total"},
 "Clinker Cooler":{"p_plus":["SECAIRNW","TAIRNW"],  "p_minus":["CLRCAIR","CLRCAIR2"],
                   "f_plus":["CLINKERC"],            "f_minus":["CLINKOUT"],
                   "f_work":False, "basis":"total"},
}

# ============================================================== COM HELPERS ===
def node(a,p):
    try: return a.Tree.FindNode(p)
    except Exception: return None
def rv(a,p):
    n=node(a,p)
    if n is None: return None
    try: return n.Value
    except Exception: return None
def num(x): return float(x) if isinstance(x,(int,float)) else 0.0

# ============================================================ READ STREAM ======
def read_stream(a, st):
    b=r"\Data\Streams\%s\Output"%st
    d={"name":st,"ss":{},"comp_mol":{},"comp_mass":{},"T":None,"P":None,"exists":False}
    for ss in SUBS:
        d["ss"][ss]={"h":rv(a,b+r"\HMX_MASS\%s"%ss),
                     "s":rv(a,b+r"\SMX_MASS\%s"%ss),
                     "T":rv(a,b+r"\TEMP_OUT\%s"%ss),
                     "m":num(rv(a,b+r"\MASSFLMX\%s"%ss))}
        if d["ss"][ss]["T"] is not None:
            d["exists"]=True
            if d["T"] is None: d["T"]=d["ss"][ss]["T"]
    d["P"]=rv(a,b+r"\PRES_OUT\MIXED") or rv(a,b+r"\PRES_OUT\CISOLID") or P0
    for ss in SUBS:
        for c in COMPONENTS+[NC_COMP]:
            mm=rv(a,b+r"\MASSFLOW\%s\%s"%(ss,c))
            nm=rv(a,b+r"\MOLEFLOW\%s\%s"%(ss,c))
            if isinstance(mm,(int,float)) and abs(mm)>1e-15:
                d["comp_mass"].setdefault(ss,{})[c]=mm
            if isinstance(nm,(int,float)) and abs(nm)>1e-15:
                d["comp_mol"].setdefault(ss,{})[c]=nm
    # Aspen native exergy (only populated if exergy calcs enabled in the model)
    d["exergyms"]={}
    for ss in SUBS:
        v=rv(a,b+r"\STRM_UPP\EXERGYMS\%s\TOTAL"%ss)
        if isinstance(v,(int,float)): d["exergyms"][ss]=v
    return d

def read_coal_ultimate(a, st):
    base=r"\Data\Streams\%s\Output\COMP_ATTR"%st
    out={}
    for k in ["CARBON","HYDROGEN","NITROGEN","SULFUR","OXYGEN","ASH"]:
        v=rv(a, base+r"\%s\ULTANAL\%s\NC"%(k, NC_COMP))
        try: out[k]=float(v)
        except Exception: out[k]=0.0
    if (out.get("CARBON") or 0) <= 1.0:
        out={"CARBON":48.0,"HYDROGEN":3.5,"NITROGEN":1.2,"SULFUR":0.45,
             "OXYGEN":8.85,"ASH":38.0}
    return out

# ====================================================== PHYSICAL EXERGY ========
def _massfrac(sd, ss):
    comps=sd["comp_mass"].get(ss,{})
    tot=sum(v for c,v in comps.items() if c!=NC_COMP)
    if tot<=0: return {}
    return {c:v/tot for c,v in comps.items() if c!=NC_COMP}

def _comp_mismatch(sd, rd, ss):
    f1=_massfrac(sd,ss); f2=_massfrac(rd,ss)
    if not f1 or not f2: return None
    keys=set(f1)|set(f2)
    return max(abs(f1.get(c,0.0)-f2.get(c,0.0)) for c in keys)

def phys_exergy_refstream(st, streams):
    """Physical exergy of MIXED+CISOLID substreams via dead-state reference
       stream: ex = m[(h-h0) - T0(s-s0)] per substream. Returns (ex, note) or
       (None, reason) if the reference is unavailable -> caller falls back."""
    sd=streams.get(st)
    if not sd: return None,"no stream"
    ref=DS_REF.get(st)
    if not ref: return None,"no DS map entry"
    rd=streams.get(ref)
    if not rd or not rd.get("exists"): return None,"DS stream %s absent"%ref
    rT=rd["T"]
    if rT is None or abs(num(rT)-T0)>2.0:
        return None,"DS stream %s not at T0 (T=%s)"%(ref,rT)
    ex=0.0; used=[]; warn=""
    for ss in ("MIXED","CISOLID"):
        g=sd["ss"][ss]; r=rd["ss"][ss]
        if g["m"]<=1e-12: continue
        if g["h"] is None or g["s"] is None: return None,"%s h/s missing"%ss
        if r["h"] is None or r["s"] is None: return None,"DS %s h/s missing"%ss
        mis=_comp_mismatch(sd,rd,ss)
        if mis is not None and mis>COMP_TOL:
            warn=" COMPWARN(%.3f)"%mis
        ex+=g["m"]*((g["h"]-r["h"]) - T0*(g["s"]-r["s"]))
        used.append(ss)
    if not used: return None,"no MIXED/CISOLID mass"
    return ex, "ref=%s[%s]%s"%(ref,"+".join(used),warn)

# ---------------- fallback (previous Cp method, tagged FALLBACK) ---------------
def fb_solid(st, streams):
    sd=streams.get(st)
    if not sd: return 0.0
    cis=sd["ss"]["CISOLID"]; m=cis["m"]; T=cis["T"]
    if m<=1e-12 or T is None or T<=T0: return 0.0
    comps=sd["comp_mass"].get("CISOLID",{})
    tot=sum(v for c,v in comps.items() if c in MINERAL_CP)
    Cp=sum((comps.get(c,0)/tot)*MINERAL_CP[c] for c in MINERAL_CP) if tot>0 else DEFAULT_SOLID_CP
    return m*Cp*((T-T0)-T0*math.log(T/T0))

def fb_gas(st, streams):
    sd=streams.get(st)
    if not sd: return 0.0
    g=sd["ss"]["MIXED"]
    if g["m"]<=1e-12 or g["T"] is None or g["T"]<=T0: return 0.0
    comps=sd["comp_mass"].get("MIXED",{})
    tot=sum(v for c,v in comps.items() if c in GAS_CP_SPECIES)
    Cp=sum((comps.get(c,0)/tot)*GAS_CP_SPECIES[c] for c in GAS_CP_SPECIES if c in comps) if tot>0 else 1050.0
    if not (400.0<=Cp<=1400.0): Cp=1050.0
    T=g["T"]; ex=g["m"]*Cp*((T-T0)-T0*math.log(T/T0))
    P=num(sd["P"]) or P0
    if P>P0: ex+=g["m"]*(R/28.96e-3)*T0*math.log(P/P0)
    return max(0.0,ex)

def coal_phys_exergy(st, streams):
    sd=streams.get(st)
    if not sd: return 0.0
    nc=sd["ss"]["NC"]
    if nc["m"]<=1e-12 or nc["T"] is None or nc["T"]<=T0: return 0.0
    return nc["m"]*COAL_CP*((nc["T"]-T0)-T0*math.log(nc["T"]/T0))

# ====================================================== CHEMICAL EXERGY ========
def coal_beta(ult, verbose=False):
    """Szargut-Styrylska, corrected piecewise form (Annexure A3).
       Mass-ratio basis; ratios are basis-free (% or fraction)."""
    C=ult.get("CARBON") or 0;H=ult.get("HYDROGEN") or 0
    O=ult.get("OXYGEN") or 0;N=ult.get("NITROGEN") or 0;S=ult.get("SULFUR") or 0
    if C<=0: return 0.0
    HC=H/C;OC=O/C;NC=N/C
    if OC<=0.667:
        beta=1.0437+0.1882*HC+0.0610*OC+0.0404*NC
        branch="linear (O/C<=0.667)"
    else:
        beta=(1.0438+0.1882*HC-0.2509*(1+0.7256*HC)+0.0383*NC)/(1-0.3035*OC)
        branch="biomass (0.667<O/C<=2.67)"
    # basis detection from carbon: ultimate analysis in percent if C > 1
    Sfrac = S/100.0 if C > 1.0 else S
    ncv=_NCV_ACTIVE[0]
    ex=beta*ncv+EX_SULFUR*Sfrac
    if verbose:
        print("    [coal] C=%.2f H=%.2f O=%.2f N=%.2f S=%.2f  O/C=%.3f %s"
              %(C,H,O,N,S,OC,branch))
        print("    [coal] beta=%.4f  NCV_dry=%.0f kJ/kg  ->  ex=%.2f MJ/kg"
              %(beta,ncv/1e3,ex/1e6))
    return ex

def coal_element_sums(ult):
    """(sum nu*s_el [J/kg-K], sum nu*ex_el [J/kg]) for dry coal from ultanal %."""
    f=lambda k: (ult.get(k) or 0.0)/100.0
    n={"C":f("CARBON")/0.012011,"H2":f("HYDROGEN")/0.002016,
       "O2":f("OXYGEN")/0.031999,"N2":f("NITROGEN")/0.028014,
       "S":f("SULFUR")/0.032066}
    return (sum(n[k]*S_EL[k] for k in n), sum(n[k]*EX_EL[k] for k in n))

def coal_ex_consistent():
    """J/kg; None if model hf or element sums unavailable."""
    if _HF_COAL[0] is None or _COAL_ELS[0] is None: return None
    Ssum,EXsum=_COAL_ELS[0]
    return _HF_COAL[0] - T0*(COAL_S0-Ssum) + EXsum

def coal_s_conv(T=T0):
    """Coal entropy in Aspen's formation convention [J/kg-K] at T."""
    Ssum = _COAL_ELS[0][0] if _COAL_ELS[0] else 0.0
    return (COAL_S0 - Ssum) + COAL_CP*math.log(max(T,1.0)/T0)

def chemical_exergy(a, sd):
    """sum n_i*ex0_i (+ RT0 sum x ln x for MIXED only) + NC coal correlation.
       MOLEFLOW is kmol/s; ex0 kJ/mol -> W = n*1e6*ex0."""
    ex=0.0
    for ss in SUBS:
        comps=sd["comp_mol"].get(ss,{})
        for c,nf in comps.items():
            if c==NC_COMP: continue
            e0=CHEM_EX.get(c)
            if e0 is None: continue
            ex+=nf*e0*1.0e6
        if ss=="MIXED":
            ntot=sum(comps.values())
            if ntot>0:
                for c,nf in comps.items():
                    x=nf/ntot
                    if x>0: ex+=nf*1000.0*R*T0*math.log(x)
    for ss in SUBS:
        mnc=sd["comp_mass"].get(ss,{}).get(NC_COMP)
        if mnc and mnc>1e-12:
            exc = coal_ex_consistent() if USE_CONSISTENT_COAL_EX else None
            if exc is None:
                exc = coal_beta(read_coal_ultimate(a, sd["name"]))
            ex+=mnc*exc
    return ex

# ============================================================ BLOCK / ENTROPY ==
DUTY_NODE_OVERRIDE = {"KLNLOSS": ["QCALC","QNET","Q"]}   # blocks whose net
# duty reads 0 because the duty leaves through an attached heat stream that
# crosses the section boundary; for these, the calculated duty is the
# boundary heat. (The per-section energy-closure check catches any others.)

def block_duty(a,blk):
    """Net duty [W]; QNET first (accounts for attached heat streams),
       unless overridden per block."""
    for kw in DUTY_NODE_OVERRIDE.get(blk, ["QNET","QCALC","Q"]):
        v=rv(a,r"\Data\Blocks\%s\Output\%s"%(blk,kw))
        if isinstance(v,(int,float)): return v
    return 0.0

def block_T(a,blk):
    """Block outlet temperature [K]; falls back to FALLBACK_T, then T0."""
    for kw in ["B_TEMP","TEMP_OUT","TOUT"]:
        v=rv(a,r"\Data\Blocks\%s\Output\%s"%(blk,kw))
        if isinstance(v,(int,float)) and v>0: return float(v)
    return FALLBACK_T.get(blk, T0)

def model_ncv(a, streams):
    """Derive the NET calorific value [J/kg] actually embedded in the model's
    HCOALGEN enthalpy, so fuel energy and fuel exergy share one basis:
        hf_coal(298K) = NC mass enthalpy of coal at 298.15 K
        NCV = hf_coal - sum(nu_i * dHf_products,i)   (H2O vapour, S->SO2)
    Uses CM-INLET (already at 298.15 K) if present, else CALCOAL - Cp*dT."""
    hf=None
    sd=streams.get("CM-INLET")
    if sd and sd["ss"]["NC"]["h"] is not None and abs(num(sd["ss"]["NC"]["T"])-T0)<2.0:
        hf=sd["ss"]["NC"]["h"]
    else:
        sd=streams.get("CALCOAL")
        if sd and sd["ss"]["NC"]["h"] is not None:
            hf=sd["ss"]["NC"]["h"]-COAL_CP*(num(sd["ss"]["NC"]["T"])-T0)
    if hf is None: return None
    u=read_coal_ultimate(a,"CALCOAL")
    C=u["CARBON"]/100.0; H=u["HYDROGEN"]/100.0; S=u["SULFUR"]/100.0
    hf_prod = (C/0.012011)*(-393.51e3) + (H/0.002016)*(-241.83e3) \
              + (S/0.032066)*(-296.81e3)          # J per kg coal, H2O vapour
    _HF_COAL[0]=hf
    return hf - hf_prod

def entropy_flow(a, st):
    b=r"\Data\Streams\%s\Output"%st
    S=0.0
    for ss in SUBS:
        s=rv(a,b+r"\SMX_MASS\%s"%ss); m=rv(a,b+r"\MASSFLMX\%s"%ss)
        if isinstance(m,(int,float)) and m>1e-9:
            if isinstance(s,(int,float)): S+=s*m
            elif ss=="NC":   # Eisermann s0, converted to Aspen's
                Tn=rv(a,b+r"\TEMP_OUT\%s"%ss)   # formation-entropy convention
                Tn=float(Tn) if isinstance(Tn,(int,float)) and Tn>0 else T0
                S+=m*coal_s_conv(Tn)
    return S

# ============================================================ MAIN =============
def main():
    try: import win32com.client as win32
    except ImportError: print("pip install pywin32"); sys.exit(1)
    try: import openpyxl
    except ImportError: print("pip install openpyxl"); sys.exit(1)

    print("="*72); print("EXERGY CALCULATOR v2.2  (reference streams + live heat/work blocks)"); print("="*72)
    a=win32.Dispatch("Apwn.Document.40.0")
    try: a.SuppressDialogs=1
    except Exception: pass
    print(">> opening model: %s"%os.path.basename(BKP)); a.InitFromArchive2(BKP)
    try: a.Visible=0
    except Exception: pass
    print(">> running base case (once)...")
    try: a.Engine.Run2(); print("   done.")
    except Exception as e: print("   warn:",e)

    # gather streams: section boundaries + DS references + anchors
    all_streams=set()
    for s in SECTIONS.values():
        all_streams|=set(s["in"])|set(s["out"])
    all_streams|={"CLRCAIR","CLINKER","RM-INLET","RAWMEAL","RMEALK","RMEALC",
                  "PHK1SOL","PHC1SOL","PHK5SOL","PHC5SOL","PHK1GAS","PHC1GAS"}
    all_streams|=set(DS_REF.values())
    all_streams=sorted(all_streams)

    print("\n>> reading streams (read-only, no re-runs)...")
    streams={st:read_stream(a,st) for st in all_streams}

    missing_ds=sorted({r for r in DS_REF.values()
                       if not streams.get(r,{}).get("exists")})
    if missing_ds:
        print("\n!! DEAD-STATE REFERENCE STREAMS NOT FOUND: %s"%", ".join(missing_ds))
        print("!! Falling back to Cp-based physical exergy for affected streams.")
        print("!! Add the Dupl+Heater trains (see header) for the consistent method.\n")

    print(">> coal energy/exergy alignment:")
    ncv_m = model_ncv(a, streams)
    if ncv_m is not None:
        print("    NCV embedded in model (HCOALGEN) = %.0f kJ/kg  (config NCV_DRY = %.0f)"
              %(ncv_m/1e3, NCV_DRY/1e3))
        if abs(ncv_m-NCV_DRY)>0.2e6:
            print("    !! model and config differ by %.2f MJ/kg"%((ncv_m-NCV_DRY)/1e6))
        if USE_MODEL_NCV:
            _NCV_ACTIVE[0]=ncv_m
            print("    -> USING MODEL NCV for Szargut coal exergy (USE_MODEL_NCV=True)")
        else:
            _NCV_ACTIVE[0]=NCV_DRY
    else:
        _NCV_ACTIVE[0]=NCV_DRY
        print("    (could not derive model NCV; using config NCV_DRY)")

    print("\n>> coal check (CALCOAL ultimate analysis):")
    try:
        u=read_coal_ultimate(a,"CALCOAL")
        _COAL_ELS[0]=coal_element_sums(u)
        coal_beta(u, verbose=True)
        exc=coal_ex_consistent()
        if exc is not None:
            print("    [coal] element-consistent ex (compound formula, model hf)"
                  " = %.2f MJ/kg%s"%(exc/1e6,
                  "  <- USED" if USE_CONSISTENT_COAL_EX else ""))
    except Exception: pass

    print("\n>> computing per-stream exergy...\n")
    SR={}
    for st in all_streams:
        if st in set(DS_REF.values())-set(DS_REF.keys()) and st not in DS_REF:
            continue  # pure reference streams are not reported
        sd=streams[st]
        mtot=sum(sd["ss"][ss]["m"] for ss in SUBS)
        ex_ph_ms, note = phys_exergy_refstream(st, streams)
        if ex_ph_ms is None:
            nc_only = (sd["ss"]["NC"]["m"]>1e-12 and
                       sd["ss"]["MIXED"]["m"]<=1e-12 and
                       sd["ss"]["CISOLID"]["m"]<=1e-12)
            ex_ph_ms = fb_solid(st,streams)+fb_gas(st,streams)
            note = "NC closed-form" if nc_only else "FALLBACK(%s)"%note
        ex_c  = coal_phys_exergy(st, streams)
        ex_ph = ex_ph_ms+ex_c
        ex_ch = chemical_exergy(a, sd)
        aspen_ex = sum(v*sd["ss"][ss]["m"] for ss,v in sd["exergyms"].items()
                       if sd["ss"][ss]["m"]>1e-12) if sd["exergyms"] else None
        SR[st]={"ex_ph":ex_ph,"ex_ch":ex_ch,"ex_tot":ex_ph+ex_ch,"T":sd["T"],
                "m":mtot,"tag":note,"ph_coal":ex_c,"aspen_ex":aspen_ex,"psi":None}
        if mtot>1e-9:
            print("  %-10s T=%7.1fK  Ex_ph=%9.3f  Ex_ch=%9.3f  tot=%9.3f MW  %s"
                  %(st,num(sd["T"]),ex_ph/1e6,ex_ch/1e6,(ex_ph+ex_ch)/1e6,note))

    def EX(s):
        r=SR.get(s); return r["ex_tot"] if r else 0.0

    def PHI(s):
        """Dead-state consistency potential: H0 - T0*S0 + Ex_ch [W]."""
        sd=streams.get(s); r=SR.get(s)
        if not sd or not r: return None
        ref=DS_REF.get(s)
        rd=streams.get(ref) if ref else None
        tot=0.0
        for ss in ("MIXED","CISOLID"):
            g=sd["ss"][ss]
            if g["m"]<=1e-12: continue
            if not rd or not rd.get("exists"): return None
            h0=rd["ss"][ss]["h"]; s0=rd["ss"][ss]["s"]
            if h0 is None or s0 is None: return None
            tot+=g["m"]*(h0-T0*s0)
        nc=sd["ss"]["NC"]
        if nc["m"]>1e-12:
            if _HF_COAL[0] is None or _COAL_ELS[0] is None: return None
            tot+=nc["m"]*(_HF_COAL[0]-T0*coal_s_conv(T0))
        return tot-r["ex_ch"]          # Psi = G0 - Ex_ch

    for st in list(SR):
        try: SR[st]["psi"]=PHI(st)
        except Exception: SR[st]["psi"]=None

    # ---------------- sections: Route B primary, Route A cross-check ----------
    print("\n"+"="*72); print("SECTION RESULTS"); print("="*72)
    S_cache={}
    def S_of(st):
        if st not in S_cache: S_cache[st]=entropy_flow(a,st)
        return S_cache[st]

    SEC={}; FLAGS=[]; BLKINFO=[]
    def H_of(st):
        sd=streams.get(st)
        if not sd: return 0.0
        Ht=0.0
        for ss in SUBS:
            g=sd["ss"][ss]
            if g["m"]>1e-12 and g["h"] is not None: Ht+=g["m"]*g["h"]
        return Ht

    for sec,m in SECTIONS.items():
        ex_in =sum(EX(s) for s in m["in"])
        ex_out=sum(EX(s) for s in m["out"])
        # ---- heat & work blocks (live duties + outlet temperatures) ----
        blks=[]
        for blk,kind in SECTION_BLOCKS.get(sec,[]):
            Q=block_duty(a,blk); Tb=block_T(a,blk)
            if kind=="work" and Q>0:
                blks.append([blk,"work",Q,Tb,Q])
            else:
                qx=Q*(1.0-T0/Tb) if Tb>0 else 0.0
                blks.append([blk,"heat",Q,Tb,qx])
        # ---- per-section ENERGY closure & auto-pairing -------------------
        # If a block's duty is supplied INTERNALLY through an attached heat
        # stream (e.g. Q-DECOMP: RYield <-> RGibbs), Aspen still reports the
        # duty on its node, but no energy crosses the section boundary. The
        # closure residual then equals exactly minus that duty; detect this
        # and reclassify the block as internal (excluded from both routes).
        dH = sum(H_of(s) for s in m["out"]) - sum(H_of(s) for s in m["in"])
        def _resid():
            return dH - sum(b[2] for b in blks if b[1] in ("work","heat"))
        E_resid=_resid()
        if abs(E_resid)>ENERGY_CLOSURE_TOL:
            for b in blks:
                if b[1]=="heat" and abs(E_resid + b[2])<ENERGY_CLOSURE_TOL:
                    print("    [auto] %s duty %.2f MW is internal (heat-stream"
                          " paired) - excluded from boundary"%(b[0],b[2]/1e6))
                    b[1]="internal"; b[4]=0.0
                    E_resid=_resid()
                    break
        if abs(E_resid)>ENERGY_CLOSURE_TOL:
            FLAGS.append("%s: ENERGY balance open by %.2f MW -> boundary stream/block missing"
                         %(sec,E_resid/1e6))
        W  = sum(b[2] for b in blks if b[1]=="work")
        Qx = sum(b[4] for b in blks if b[1]=="heat")
        SQ = sum(b[2]/b[3] for b in blks if b[1]=="heat" and b[3]>0)
        for b in blks: BLKINFO.append((sec,b[0],b[1],b[2],b[3],b[4]))
        # Route B: exergy balance (work + boundary heat enter as exergy)
        I_B = ex_in + W + Qx - ex_out
        # Route A: Gouy-Stodola with boundary-heat entropy
        S_gen = sum(S_of(s) for s in m["out"]) - sum(S_of(s) for s in m["in"]) - SQ
        I_A = T0*S_gen
        resid = (I_A-I_B)/abs(I_B)*100 if abs(I_B)>1e3 else float("nan")
        # thermochemical-data consistency: predicted A-B residual
        phis_in =[PHI(s) for s in m["in"]];  phis_out=[PHI(s) for s in m["out"]]
        if all(p is not None for p in phis_in+phis_out):
            phi_gap = sum(phis_in)-sum(phis_out)
        else:
            phi_gap = None
        # Kotas functional efficiency (basis-aware: "total" or "ph")
        d=EFF_DEF[sec]
        basis=d.get("basis","total")
        if basis=="ph":
            val=lambda s: SR[s]["ex_ph"] if s in SR else 0.0
        else:
            val=EX
        p_num = sum(val(s) for s in d["p_plus"]) - sum(val(s) for s in d["p_minus"])
        f_den = sum(val(s) for s in d["f_plus"]) - sum(val(s) for s in d["f_minus"]) \
                + (W if d["f_work"] else 0.0)
        eps_f = p_num/f_den if f_den>0 else float("nan")
        eps_g = ex_out/ex_in if ex_in>0 else float("nan")
        if I_B<0: FLAGS.append("%s: Route-B irreversibility NEGATIVE (%.2f MW) -> data inconsistency"%(sec,I_B/1e6))
        if p_num<0: FLAGS.append("%s: functional product exergy gain NEGATIVE (%.2f MW) -> revisit product definition or chemistry data"%(sec,p_num/1e6))
        if eps_f==eps_f and eps_f>1.0: FLAGS.append("%s: functional efficiency > 100%% (%.1f%%)"%(sec,eps_f*100))
        eps_ip = min(max(eps_f,0.0),1.0) if eps_f==eps_f else 0.0
        IP=(1.0-eps_ip)*max(I_B,0.0)
        SEC[sec]={"ex_in":ex_in,"ex_out":ex_out,"W":W,"Qx":Qx,"E_resid":E_resid,
                  "I_B":I_B,"I_A":I_A,"resid":resid,
                  "eps_f":eps_f,"eps_g":eps_g,"IP":IP,"S_gen":S_gen,"basis":basis}
        print("\n  %-16s Ex_in=%8.3f  Ex_out=%8.3f  W=%6.3f  Q_ex=%8.3f MW  [E-closure %+0.3f MW]"
              %(sec,ex_in/1e6,ex_out/1e6,W/1e6,Qx/1e6,E_resid/1e6))
        print("    I RouteB(balance)=%9.3f MW | RouteA(T0*Sgen)=%9.3f MW | resid=%6.1f %%"
              %(I_B/1e6,I_A/1e6,resid))
        if phi_gap is not None:
            print("    [data] tables-vs-databank gap predicts A-B residual %+.3f MW"
                  " (actual %+.3f MW)"%(phi_gap/1e6,(I_A-I_B)/1e6))
        print("    eps_functional[%s] = %7.2f %%   eps_gross = %7.2f %%   IP = %8.3f MW"
              %(basis,
                eps_f*100 if eps_f==eps_f else float('nan'),
                eps_g*100 if eps_g==eps_g else float('nan'),IP/1e6))

    # ---------------- plant ----------------
    tot_IB=sum(s["I_B"] for s in SEC.values())
    tot_IA=sum(s["I_A"] for s in SEC.values())
    tot_IP=sum(s["IP"]  for s in SEC.values())
    coal_ex=EX("CALCOAL")+EX("KILNCOAL")
    elec=sum(q for (_,_,k,q,_,_) in BLKINFO if k=="work")
    Qx_all=sum(qx for (_,_,k,_,_,qx) in BLKINFO if k=="heat")
    clk=EX("CLINKOUT")
    epsP=clk/(coal_ex+elec) if (coal_ex+elec)>0 else 0
    # plant-boundary Route B closure (all boundary heat included)
    pin = (EX("RM-INLET")+EX("RMINLETG")+EX("CM-INLET")+EX("CMINLETG")
           +EX("CLRCAIR")+EX("CLRCAIR2")+elec)
    pout= (EX("CLINKOUT")+EX("RMEXHUST")+EX("CMEXHUST")
           +EX("PHK1GAS")+EX("PHC1GAS")+EX("CLREXH"))
    I_plant_boundary = pin + Qx_all - pout
    print("\n"+"="*72); print("PLANT TOTALS"); print("="*72)
    print("  Sum of section I (Route B)    = %8.3f MW"%(tot_IB/1e6))
    print("  Sum of section I (Route A)    = %8.3f MW"%(tot_IA/1e6))
    print("  Plant-boundary balance I      = %8.3f MW  (top-level closure)"%(I_plant_boundary/1e6))
    print("  Total improvement potential   = %8.3f MW"%(tot_IP/1e6))
    print("  Coal chemical exergy in       = %8.3f MW"%(coal_ex/1e6))
    print("  Electrical work in            = %8.3f MW"%(elec/1e6))
    print("  Clinker product exergy        = %8.3f MW"%(clk/1e6))
    print("  Overall plant efficiency      = %8.2f %%"%(epsP*100))
    if FLAGS:
        print("\n  !! VALIDATION FLAGS:")
        for f in FLAGS: print("     - "+f)

    write_workbook(openpyxl, SR, SEC,
        {"tot_IB":tot_IB,"tot_IA":tot_IA,"tot_IP":tot_IP,"coal_ex":coal_ex,
         "elec":elec,"clk":clk,"epsP":epsP,"I_pb":I_plant_boundary,
         "flags":FLAGS,"missing_ds":missing_ds,"blkinfo":BLKINFO,
         "ncv":_NCV_ACTIVE[0]})
    print("\nWorkbook -> %s"%OUT_XLSX)
    try: a.Close()
    except Exception: pass
    try: a.Quit()
    except Exception: pass
    print("Done. Model unchanged.")

# ============================================================ WORKBOOK =========
def write_workbook(openpyxl, SR, SEC, P):
    wb=openpyxl.Workbook()
    ws=wb.active; ws.title="Stream Exergy"
    ws.append(["Stream","T [K]","mdot [kg/s]","Ex_phys [MW]","Ex_chem [MW]",
               "Ex_total [MW]","Psi=G0-ExCh [MW]","ph_coal [MW]",
               "Aspen EXERGYMS [MW]","method"])
    for st in sorted(SR):
        r=SR[st]
        ws.append([st,round(num(r["T"]),2),round(r["m"],4),
                   round(r["ex_ph"]/1e6,4),round(r["ex_ch"]/1e6,4),
                   round(r["ex_tot"]/1e6,4),
                   round(r["psi"]/1e6,4) if r.get("psi") is not None else "",
                   round(r["ph_coal"]/1e6,6),
                   round(r["aspen_ex"]/1e6,4) if r["aspen_ex"] is not None else "",
                   r["tag"] or ""])
    ws2=wb.create_sheet("Section Exergy")
    ws2.append(["Section","Ex_in [MW]","Ex_out [MW]","Work [MW]","Q-exergy [MW]",
                "Energy closure [MW]",
                "I RouteB balance [MW]","I RouteA T0*Sgen [MW]","Residual A-B [%]",
                "eps functional [%]","eps basis","eps gross out/in [%]","IP [MW]"])
    for sec,s in SEC.items():
        ws2.append([sec,round(s["ex_in"]/1e6,3),round(s["ex_out"]/1e6,3),
                    round(s["W"]/1e6,3),round(s["Qx"]/1e6,3),
                    round(s.get("E_resid",0.0)/1e6,3),
                    round(s["I_B"]/1e6,3),round(s["I_A"]/1e6,3),
                    round(s["resid"],1) if s["resid"]==s["resid"] else "",
                    round(s["eps_f"]*100,2) if s["eps_f"]==s["eps_f"] else "",
                    s.get("basis","total"),
                    round(s["eps_g"]*100,2) if s["eps_g"]==s["eps_g"] else "",
                    round(s["IP"]/1e6,3)])
    ws3=wb.create_sheet("Plant Total")
    for kv in [("Sum section I, Route B [MW]",P["tot_IB"]/1e6),
               ("Sum section I, Route A [MW]",P["tot_IA"]/1e6),
               ("Plant-boundary balance I [MW]",P["I_pb"]/1e6),
               ("Total improvement potential [MW]",P["tot_IP"]/1e6),
               ("Coal chemical exergy in [MW]",P["coal_ex"]/1e6),
               ("Electrical work in [MW]",P["elec"]/1e6),
               ("Clinker product exergy [MW]",P["clk"]/1e6),
               ("Overall plant efficiency [%]",P["epsP"]*100)]:
        ws3.append([kv[0],round(kv[1],3)])
    wsv=wb.create_sheet("Validation")
    wsv.append(["Check","Status"])
    wsv.append(["Dead-state reference streams present",
                "ALL" if not P["missing_ds"] else "MISSING: "+", ".join(P["missing_ds"])])
    wsv.append(["Route A vs Route B residual target","< ~2 % per section"])
    for f in P["flags"]: wsv.append(["FLAG",f])
    wsb=wb.create_sheet("Heat-Work Blocks")
    wsb.append(["Section","Block","Kind","Q net [MW]","T_out [K]","Exergy of Q [MW]"])
    for (sec,blk,kind,q,tb,qx) in P.get("blkinfo",[]):
        wsb.append([sec,blk,kind,round(q/1e6,3),round(tb,1),round(qx/1e6,3)])
    ws3.append(["NCV used for coal exergy [kJ/kg]",round(P.get("ncv",0)/1e3,1)])
    a1=wb.create_sheet("A1 Conventional ExCh")
    a1.append(["Aspen ID","Species","ex0_ch [kJ/mol]","Source"])
    for aid,sp,v in [("CARBO-01","CO2",19.87),("OXYGE-01","O2",3.97),
        ("WATER","H2O(v)",9.50),("(H2O liq)","H2O(l)",0.90),("NITRO-01","N2",0.72),
        ("CARBO-02","CO",275.10),("NITRI-01","NO",88.90),("SULFU-01","SO2",313.40),
        ("CALCI-01","CaCO3",1.00),("CALCI-02","CaO",110.20),("SILIC-01","SiO2",1.90),
        ("ALUMI-01","Al2O3",200.40),("HEMAT-01","Fe2O3",16.50),("MAGNE-01","MgCO3",37.90),
        ("MAGNE-02","MgO",66.80),("C","C(graphite)",410.26),("H2","H2",236.10),
        ("COAL-ASH","ash",0.00)]:
        a1.append([aid,sp,v,"Szargut RE Model II (1988)"])
    a2=wb.create_sheet("A2 Clinker ExCh")
    a2.append(["Aspen ID","Mineral","Oxide formula","dG_f,ox [kJ/mol]","ex0_ch [kJ/mol]","Source"])
    a2.append(["TRICA-01","C3S alite","3CaO.SiO2",DG_OX["TRICA-01"],CHEM_EX["TRICA-01"],"Hanein et al. 2020 (Annex A2)"])
    a2.append(["OLIVI-01","C2S belite","2CaO.SiO2",DG_OX["OLIVI-01"],CHEM_EX["OLIVI-01"],"Hanein et al. 2020 (Annex A2)"])
    a2.append(["(CAO)-01","C3A aluminate","3CaO.Al2O3",DG_OX["(CAO)-01"],CHEM_EX["(CAO)-01"],"Hanein et al. 2020 (Annex A2)"])
    a2.append(["C4AF","brownmillerite","4CaO.Al2O3.Fe2O3",DG_OX["C4AF"],CHEM_EX["C4AF"],"Thorvaldson 1938/Zhu 2011 via Hanein 2020"])
    a3=wb.create_sheet("A3 Coal ExCh")
    a3.append(["Parameter","Value","Note"])
    a3.append(["Correlation","beta*NCV_dry + 9417*S","Szargut-Styrylska, piecewise (Annex A3)"])
    a3.append(["O/C<=0.667 branch","1.0437+0.1882(H/C)+0.0610(O/C)+0.0404(N/C)","applies to this coal (O/C=0.184)"])
    a3.append(["NCV_dry [J/kg]",NCV_DRY,"must be NET CV; see Annexure A3 note"])
    a3.append(["Coal Cp [J/kg-K]",COAL_CP,"two-point CM-INLET/CMDRYCOL"])
    a3.append(["Coal s0 [J/kg-K]",COAL_S0,"Eisermann 1980; Route-A check only"])
    mth=wb.create_sheet("Method")
    for L in [
      "Dead state T0=298.15 K, P0=101325 Pa",
      "Ex_ph (MIXED+CISOLID) = sum_ss m[(h-h0)-T0(s-s0)], h0/s0 from dead-state",
      "  reference streams (Dupl+Heater at 25C/1.01325 bar) - Hinderink 1996",
      "Ex_ph (NC coal) = m*Cp*[(T-T0)-T0 ln(T/T0)], Cp=1134 (no NC entropy in Aspen)",
      "Ex_ch: Szargut RE-II + clinker phases per Hanein 2020 + coal Szargut-Styrylska",
      "Mixing term RT0 sum(x ln x): MIXED substream only",
      "I Route B = sum Ex_in + W + sum Q(1-T0/Tk) - sum Ex_out  (primary)",
      "I Route A = T0*S_gen (Aspen entropies; coal entropy Eisermann)  (cross-check)",
      "eps = Kotas functional Ex_product/Ex_fuel per section; IP=(1-eps)*I_B",
      "No abs() guards, no efficiency caps: negatives are flagged, not hidden",
    ]: mth.append([L])
    wb.save(OUT_XLSX)

if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
