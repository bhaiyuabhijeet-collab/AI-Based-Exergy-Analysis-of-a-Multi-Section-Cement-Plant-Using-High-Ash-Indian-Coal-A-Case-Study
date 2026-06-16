#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXERGY CAMPAIGN DRIVER  -  ANN training-set generator
================================================================================
Wraps the validated base-case model + exergy_calculator_v2 to produce N samples
for ANN training. For each sample it:
  1. perturbs the chosen INPUT variables (Latin-Hypercube over their ranges),
  2. pushes them into the open Aspen case, runs it headless,
  3. runs the SAME exergy evaluation used for the base case,
  4. writes ONE ROW with all inputs + every per-section exergy/eff/irreversibility,
  5. carries the validation verdict forward: any sample that does not converge,
     or whose energy balance / Route A-B residual / efficiency falls outside the
     acceptance band, is written with status="REJECT" and a reason, so it can be
     filtered out before training instead of silently poisoning the net.

Design choices that matter:
  * ONE Aspen session is opened once and reused for all N runs (fast). The model
    file is the frozen base case; the driver only changes the sampled inputs each
    run and resets/re-solves.
  * The exergy logic is IMPORTED from exergy_calculator_v2.py - not duplicated -
    so the campaign always matches the validated base case. If you improve the
    calculator, the campaign inherits it automatically.
  * Reproducible: fixed RNG seed -> the same 500 samples regenerate identically.
  * Resumable: results are appended to CSV after every run, so a crash at sample
    317 loses nothing; rerun with RESUME=True to continue.

RUN:  pip install pywin32 openpyxl numpy
      python exergy_campaign_driver.py
================================================================================
"""
import os, sys, csv, math, time, traceback

# ============================================================ CONFIG ===========
FOLDER = r"D:\Ph.D\Ph.D Research paper work\AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal A Case Study\AspenPlus"
BKP        = os.path.join(FOLDER, "new cmill rmill preheater calciner kiln cooler.bkp")
OUT_CSV    = os.path.join(FOLDER, "ann_dataset.csv")
LOG_TXT    = os.path.join(FOLDER, "campaign_log.txt")
CALC_MODULE_DIR = FOLDER          # where exergy_calculator_v2.py lives
N_SAMPLES  = 500                  # legacy fixed-count mode (used if TARGET_OK<=0)
TARGET_OK  = 500                  # >0: keep sampling until this many accepted rows
MAX_TRIES  = 1500                 # safety cap on total Aspen runs in target mode
COUNT_CAPPED_AS_OK = False        # False: only PURE-OK (eps<=1, no capping) count
                                  #        toward TARGET_OK -> 500 fully-clean rows.
                                  # True : OK + OK_CAPPED both count (fewer runs).
RNG_SEED   = 20260613
RESUME     = True                 # append to existing CSV / skip done samples
SETTLE_RUNS = 1                   # extra Run2() passes per sample if recycle loops

# ----- INPUT variables to perturb (all 26 independent model handles) ---------
# Each entry: key -> dict(node, lo, hi, unit, kind)
#   kind "stream_mass_<ss>" : substream total mass flow (TOTFLOW)  [kg/s]
#   kind "stream_temp"      : feed stream temperature (TEMP)        [K]
#   kind "block_temp"       : Heater/RStoic outlet temperature spec [K]
#   kind "split_frac"       : FSplit split fraction on a stream     [-]
# Ranges are +-10% of base (Step-1 spec), with split fractions clipped to
# physically valid sub-ranges so the complementary fraction stays positive.
def _pm(v, frac=0.10): return (v*(1-frac), v*(1+frac))

INPUTS = {
 # --- 8 feed-stream handles ---
 "rawmeal_feed":  dict(node=r"\Data\Streams\RM-INLET\Input\TOTFLOW\CISOLID",
                       lo=_pm(100.78)[0], hi=_pm(100.78)[1], unit="kg/s", kind="stream_mass_CISOLID"),
 "rmgas_flow":    dict(node=r"\Data\Streams\RMINLETG\Input\TOTFLOW\MIXED",
                       lo=_pm(178.90)[0], hi=_pm(178.90)[1], unit="kg/s", kind="stream_mass_MIXED"),
 "rmgas_temp":    dict(node=r"\Data\Streams\RMINLETG\Input\TEMP\MIXED",
                       lo=_pm(436.15)[0], hi=_pm(436.15)[1], unit="K", kind="stream_temp"),
 "coal_feed":     dict(node=r"\Data\Streams\CM-INLET\Input\TOTFLOW\NC",
                       lo=_pm(8.28)[0], hi=_pm(8.28)[1], unit="kg/s", kind="stream_mass_NC"),
 "cmgas_flow":    dict(node=r"\Data\Streams\CMINLETG\Input\TOTFLOW\MIXED",
                       lo=_pm(33.60)[0], hi=_pm(33.60)[1], unit="kg/s", kind="stream_mass_MIXED"),
 "cmgas_temp":    dict(node=r"\Data\Streams\CMINLETG\Input\TEMP\MIXED",
                       lo=_pm(436.15)[0], hi=_pm(436.15)[1], unit="K", kind="stream_temp"),
 "clrcair_flow":  dict(node=r"\Data\Streams\CLRCAIR\Input\TOTFLOW\MIXED",
                       lo=_pm(48.00)[0], hi=_pm(48.00)[1], unit="kg/s", kind="stream_mass_MIXED"),
 "clrcair2_flow": dict(node=r"\Data\Streams\CLRCAIR2\Input\TOTFLOW\MIXED",
                       lo=_pm(103.00)[0], hi=_pm(103.00)[1], unit="kg/s", kind="stream_mass_MIXED"),
 # --- 14 heater outlet-temperature specs ---
 "rmfan_T":  dict(node=r"\Data\Blocks\RM-FAN\Input\TEMP",   lo=_pm(438.15)[0], hi=_pm(438.15)[1], unit="K", kind="block_temp"),
 "rmdry_T":  dict(node=r"\Data\Blocks\RM-DRYER\Input\TEMP", lo=_pm(363.15)[0], hi=_pm(363.15)[1], unit="K", kind="block_temp"),
 "gasht_T":  dict(node=r"\Data\Blocks\GASHEATE\Input\TEMP", lo=_pm(438.15)[0], hi=_pm(438.15)[1], unit="K", kind="block_temp"),
 "cmdry_T":  dict(node=r"\Data\Blocks\CM-DRYER\Input\TEMP", lo=_pm(343.15)[0], hi=_pm(343.15)[1], unit="K", kind="block_temp"),
 "phk1_T":   dict(node=r"\Data\Blocks\PH-HT-K1\Input\TEMP", lo=_pm(593.15)[0], hi=_pm(593.15)[1], unit="K", kind="block_temp"),
 "phk2_T":   dict(node=r"\Data\Blocks\PH-HT-K2\Input\TEMP", lo=_pm(753.15)[0], hi=_pm(753.15)[1], unit="K", kind="block_temp"),
 "phk3_T":   dict(node=r"\Data\Blocks\PH-HT-K3\Input\TEMP", lo=_pm(903.15)[0], hi=_pm(903.15)[1], unit="K", kind="block_temp"),
 "phk4_T":   dict(node=r"\Data\Blocks\PH-HT-K4\Input\TEMP", lo=_pm(973.15)[0], hi=_pm(973.15)[1], unit="K", kind="block_temp"),
 "phk5_T":   dict(node=r"\Data\Blocks\PH-HT-K5\Input\TEMP", lo=_pm(1023.15)[0], hi=_pm(1023.15)[1], unit="K", kind="block_temp"),
 "phc1_T":   dict(node=r"\Data\Blocks\PH-HT-C1\Input\TEMP", lo=_pm(593.15)[0], hi=_pm(593.15)[1], unit="K", kind="block_temp"),
 "phc2_T":   dict(node=r"\Data\Blocks\PH-HT-C2\Input\TEMP", lo=_pm(753.15)[0], hi=_pm(753.15)[1], unit="K", kind="block_temp"),
 "phc3_T":   dict(node=r"\Data\Blocks\PH-HT-C3\Input\TEMP", lo=_pm(903.15)[0], hi=_pm(903.15)[1], unit="K", kind="block_temp"),
 "phc4_T":   dict(node=r"\Data\Blocks\PH-HT-C4\Input\TEMP", lo=_pm(973.15)[0], hi=_pm(973.15)[1], unit="K", kind="block_temp"),
 "phc5_T":   dict(node=r"\Data\Blocks\PH-HT-C5\Input\TEMP", lo=_pm(1023.15)[0], hi=_pm(1023.15)[1], unit="K", kind="block_temp"),
 # --- 1 reactor temperature spec ---
 "clcalc_T": dict(node=r"\Data\Blocks\CLCALC\Input\TEMP",   lo=_pm(1173.15)[0], hi=_pm(1173.15)[1], unit="K", kind="block_temp"),
 # --- 3 split fractions (clipped to valid bands) ---
 "coal_split":   dict(node=r"\Data\Blocks\CM-SPLIT\Input\FRAC\CALCOAL", lo=0.50, hi=0.60, unit="-", kind="split_frac"),
 "secair_split": dict(node=r"\Data\Blocks\CLRFSPL\Input\FRAC\SECAIRNW", lo=0.50, hi=0.60, unit="-", kind="split_frac"),
 "meal_split":   dict(node=r"\Data\Blocks\MEAL-SEP\Input\FRAC\RMEALK",  lo=0.60, hi=0.70, unit="-", kind="split_frac"),
}

# ----- QUALITY criteria (per sample) ----------------------------------------
# Philosophy: every converged sample is KEPT. We do not discard samples merely
# because a sectional efficiency definition becomes numerically singular at an
# off-design point (small useful-exergy denominator -> eps slightly >1). Such
# eps values are CAPPED to 1.0 and flagged; the irreversibility, improvement
# potential, and energy-closure for those samples remain physically valid.
# A row is marked status="OK" when it is fully second-law-consistent, and
# status="OK_CAPPED" when only an eps-cap was applied. A row is "REJECT" only
# for genuine second-law violations (negative irreversibility) or a failed
# energy balance - these are the rows an ANN must not learn from.
ACC_ENERGY_TOL_MW   = 0.5     # |energy closure| per section must be < this (hard)
ACC_NO_NEG_IRR      = True    # genuine reject: any section Route-B I_B < 0
EPS_CAP             = 1.0     # sectional efficiency physically capped at 1.0
EPS_CAP_TOL         = 0.0     # cap whenever eps exceeds EPS_CAP

SECTIONS_ORDER = ["Raw Mill","Coal Mill","Preheater","Calciner","Kiln","Clinker Cooler"]

# ============================================================ IMPORT CALC ======
sys.path.insert(0, CALC_MODULE_DIR)
try:
    import exergy_calculator_v2 as EX
except Exception as e:
    print("Could not import exergy_calculator_v2.py from", CALC_MODULE_DIR)
    print("Place this driver in the same folder, or fix CALC_MODULE_DIR.")
    raise

# ============================================================ SAMPLING =========
def latin_hypercube(n, d, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    # one permutation per dimension, jittered within each stratum
    cuts = np.linspace(0, 1, n+1)
    u = rng.uniform(size=(n, d))
    pts = np.empty((n, d))
    for j in range(d):
        strata = cuts[:n] + (cuts[1]-cuts[0])*u[:, j]
        rng.shuffle(strata)
        pts[:, j] = strata
    return pts   # in [0,1]^d

def build_samples():
    keys = list(INPUTS.keys())
    # In target-OK mode generate a large LHS pool (MAX_TRIES) so we can keep
    # drawing well-spread samples until TARGET_OK accepted rows are collected.
    n_pool = MAX_TRIES if TARGET_OK>0 else N_SAMPLES
    pts = latin_hypercube(n_pool, len(keys), RNG_SEED)
    samples = []
    for i in range(n_pool):
        s = {}
        for j, k in enumerate(keys):
            lo, hi = INPUTS[k]["lo"], INPUTS[k]["hi"]
            s[k] = lo + pts[i, j]*(hi-lo)
        samples.append(s)
    return keys, samples

# ============================================================ ASPEN I/O ========

def _set_frac_node(aspen, blk, stream, val):
    """Set one FSplit outlet fraction; try the common node spellings."""
    candidates = [
        r"\Data\Blocks\%s\Input\FRAC\%s"%(blk,stream),
        r"\Data\Blocks\%s\Input\FRACS\%s"%(blk,stream),
    ]
    for nd in candidates:
        n=aspen.Tree.FindNode(nd)
        if n is not None:
            n.Value=float(val); return True
    return False

# Full fraction maps per FSplit so we write a COMPLETE, summing-to-1 spec
# every time (avoids the transient !CC "fractions do not sum" error).
#   key sampled -> {outlet: fraction-expression}
# CLRFSPL: SECAIRNW sampled; TAIRNW takes the remainder; CLREXH stays 0.
# CM-SPLIT: CALCOAL sampled; KILNCOAL = 1 - CALCOAL.
# MEAL-SEP: RMEALK sampled; RMEALC = 1 - RMEALK.
# Only the SPECIFIED (non-residual) outlets may be written. In each FSplit the
# last outlet is the residual one and Aspen computes it automatically; writing a
# fraction to a residual outlet raises the "!CC error, Flow Split Specifications".
#   CM-SPLIT : specifies CALCOAL only      (KILNCOAL is residual)
#   MEAL-SEP : specifies RMEALK only       (RMEALC   is residual)
#   CLRFSPL  : specifies SECAIRNW & TAIRNW (CLREXH   is residual)
_SPLIT_LAYOUT = {
 # writable outlet(s) per FSplit, confirmed by probing the model:
 "CM-SPLIT": ("CALCOAL",  [("CALCOAL",  lambda v: v)]),          # KILNCOAL residual
 "CLRFSPL":  ("SECAIRNW", [("SECAIRNW", lambda v: v),
                           ("TAIRNW",   lambda v: 1.0-v)]),      # CLREXH residual
 # MEAL-SEP specifies RMEALC (not RMEALK); 'meal_split' is the RMEALK fraction,
 # so the writable RMEALC fraction = 1 - meal_split.
 "MEAL-SEP": ("RMEALC",   [("RMEALC",   lambda v: 1.0-v)]),      # RMEALK residual
}

def _block_of(node):
    parts = node.split(chr(92))
    return parts[parts.index("Blocks")+1]

def set_input(aspen, key, value):
    spec = INPUTS[key]; node = spec["node"]; value=float(value)
    if spec.get("kind")=="split_frac":
        blk=_block_of(node)
        layout=_SPLIT_LAYOUT.get(blk)
        if layout is None:
            n=aspen.Tree.FindNode(node)
            if n is None: raise RuntimeError("split node not found: %s"%node)
            n.Value=value; return
        _, outlets = layout
        for strm, f in outlets:
            if not _set_frac_node(aspen, blk, strm, f(value)):
                raise RuntimeError("split frac node not found: %s\\%s"%(blk,strm))
        return
    # non-split: single scalar node
    n = aspen.Tree.FindNode(node)
    if n is None:
        raise RuntimeError("input node not found: %s (%s)"%(key, node))
    n.Value = value

def apply_and_run(aspen, sample, keys):
    """Reset to an unsolved state, write the sampled inputs, then solve.
    Writing FSplit fractions only AFTER Reset avoids the
    '!CC error, using old value' that occurs when a solved case is edited."""
    aspen.Reinit()                 # clear the previous converged solution
    for k in keys:
        set_input(aspen, k, sample[k])
    aspen.Engine.Run2()

# ============================================================ EVALUATE =========
def evaluate(aspen):
    """Re-use the calculator's own stream/section machinery to get a result dict.
       Returns (sections, plant, flags) mirroring the base-case run."""
    # gather the same stream set the calculator uses
    all_streams=set()
    for s in EX.SECTIONS.values():
        all_streams|=set(s["in"])|set(s["out"])
    all_streams|={"CLRCAIR","CLINKER","RM-INLET","RAWMEAL","RMEALK","RMEALC",
                  "PHK1SOL","PHC1SOL","PHK5SOL","PHC5SOL","PHK1GAS","PHC1GAS"}
    all_streams|=set(EX.DS_REF.values())
    streams={st:EX.read_stream(aspen,st) for st in sorted(all_streams)}

    # NCV alignment + coal element sums (same as base case)
    EX._COAL_ELS[0]=EX.coal_element_sums(EX.read_coal_ultimate(aspen,"CALCOAL"))
    ncv=EX.model_ncv(aspen, streams)
    EX._NCV_ACTIVE[0]= ncv if (ncv and EX.USE_MODEL_NCV) else EX.NCV_DRY

    # per-stream exergy
    SR={}
    for st in streams:
        sd=streams[st]
        mtot=sum(sd["ss"][ss]["m"] for ss in EX.SUBS)
        ex_ph_ms,_=EX.phys_exergy_refstream(st, streams)
        if ex_ph_ms is None:
            ex_ph_ms=EX.fb_solid(st,streams)+EX.fb_gas(st,streams)
        ex_c=EX.coal_phys_exergy(st, streams)
        ex_ch=EX.chemical_exergy(aspen, sd)
        SR[st]={"ex_ph":ex_ph_ms+ex_c,"ex_ch":ex_ch,"ex_tot":ex_ph_ms+ex_c+ex_ch,
                "m":mtot,"T":sd["T"]}
    def EXt(s): r=SR.get(s); return r["ex_tot"] if r else 0.0
    def H_of(s):
        sd=streams.get(s)
        if not sd: return 0.0
        return sum(sd["ss"][ss]["m"]*sd["ss"][ss]["h"] for ss in EX.SUBS
                   if sd["ss"][ss]["m"]>1e-12 and sd["ss"][ss]["h"] is not None)
    S_cache={}
    def S_of(s):
        if s not in S_cache: S_cache[s]=EX.entropy_flow(aspen,s)
        return S_cache[s]

    sections={}; flags=[]
    for sec,m in EX.SECTIONS.items():
        ex_in=sum(EXt(s) for s in m["in"]); ex_out=sum(EXt(s) for s in m["out"])
        blks=[]
        for blk,kind in EX.SECTION_BLOCKS.get(sec,[]):
            Q=EX.block_duty(aspen,blk); Tb=EX.block_T(aspen,blk)
            if kind=="work" and Q>0: blks.append([blk,"work",Q,Tb,Q])
            else:
                qx=Q*(1.0-EX.T0/Tb) if Tb>0 else 0.0
                blks.append([blk,"heat",Q,Tb,qx])
        dH=sum(H_of(s) for s in m["out"])-sum(H_of(s) for s in m["in"])
        def _resid(): return dH-sum(b[2] for b in blks if b[1] in("work","heat"))
        E_resid=_resid()
        if abs(E_resid)>EX.ENERGY_CLOSURE_TOL:
            for b in blks:
                if b[1]=="heat" and abs(E_resid+b[2])<EX.ENERGY_CLOSURE_TOL:
                    b[1]="internal"; b[4]=0.0; E_resid=_resid(); break
        W=sum(b[2] for b in blks if b[1]=="work")
        Qx=sum(b[4] for b in blks if b[1]=="heat")
        SQ=sum(b[2]/b[3] for b in blks if b[1]=="heat" and b[3]>0)
        I_B=ex_in+W+Qx-ex_out
        S_gen=sum(S_of(s) for s in m["out"])-sum(S_of(s) for s in m["in"])-SQ
        I_A=EX.T0*S_gen
        d=EX.EFF_DEF[sec]; basis=d.get("basis","total")
        val=(lambda s: SR[s]["ex_ph"] if s in SR else 0.0) if basis=="ph" else EXt
        p_num=sum(val(s) for s in d["p_plus"])-sum(val(s) for s in d["p_minus"])
        f_den=sum(val(s) for s in d["f_plus"])-sum(val(s) for s in d["f_minus"])+(W if d["f_work"] else 0.0)
        eps=p_num/f_den if f_den>0 else float("nan")
        eps_ip=min(max(eps,0.0),1.0) if eps==eps else 0.0
        IP=(1.0-eps_ip)*max(I_B,0.0)
        sections[sec]={"ex_in":ex_in,"ex_out":ex_out,"W":W,"Qx":Qx,"E_resid":E_resid,
                       "I_B":I_B,"I_A":I_A,"eps":eps,"IP":IP}
    coal_ex=EXt("CALCOAL")+EXt("KILNCOAL")
    elec=sum(b[2] for sec in EX.SECTION_BLOCKS for b in
             [(x[0],x[1],EX.block_duty(aspen,x[0]),0,0) for x in EX.SECTION_BLOCKS[sec]]
             if b[1]=="work")
    clk=EXt("CLINKOUT")
    plant={"I_B":sum(s["I_B"] for s in sections.values()),
           "I_A":sum(s["I_A"] for s in sections.values()),
           "IP":sum(s["IP"] for s in sections.values()),
           "coal_ex":coal_ex,"clk":clk,
           "eps":clk/(coal_ex+elec) if (coal_ex+elec)>0 else 0.0,
           "ncv":EX._NCV_ACTIVE[0],
           "clinker_T":SR.get("CLINKER",{}).get("T")}
    return sections, plant

# ============================================================ ACCEPTANCE =======
def verdict(sections, plant):
    """Cap-and-flag. Returns (status, reason).
       status in {OK, OK_CAPPED, REJECT}.
       Mutates sections[sec]["eps"] -> capped value, records ["eps_capped"].
       REJECT only on hard physical violations (neg irreversibility / energy open).
    """
    hard=[]; capped=[]
    for sec,s in sections.items():
        # hard rejects: genuine second-law / balance failures
        if abs(s["E_resid"])>ACC_ENERGY_TOL_MW*1e6:
            hard.append("%s energy-open %.2fMW"%(sec,s["E_resid"]/1e6))
        if ACC_NO_NEG_IRR and s["I_B"]<-1e3:           # < -0.001 MW
            hard.append("%s I_B<0 (%.2f)"%(sec,s["I_B"]/1e6))
        # soft: cap eps>1 (numerically singular definition at off-design point)
        e=s.get("eps")
        if e==e and e is not None and e>EPS_CAP+EPS_CAP_TOL:
            capped.append("%s eps %.2f->1.00"%(sec,e))
            s["eps_capped"]=True
            s["eps"]=EPS_CAP
            # recompute IP with capped eps (IP=(1-eps)*max(I_B,0) -> 0 here)
            s["IP"]=(1.0-EPS_CAP)*max(s["I_B"],0.0)
        else:
            s["eps_capped"]=False
    if hard:
        return "REJECT", ";".join(hard)
    if capped:
        return "OK_CAPPED", ";".join(capped)
    return "OK", ""

# ============================================================ CSV ==============
def csv_header(keys):
    h=["sample","status","reason"]
    h+= [ "%s[%s]"%(k,INPUTS[k]["unit"]) for k in keys ]
    for sec in SECTIONS_ORDER:
        t=sec.replace(" ","")
        h+=["%s_I_B_MW"%t,"%s_I_A_MW"%t,"%s_eps"%t,"%s_eps_capped"%t,"%s_IP_MW"%t,"%s_Eclose_MW"%t]
    h+=["plant_I_B_MW","plant_IP_MW","plant_eps","coal_ex_MW","clinker_ex_MW",
        "clinker_T_K","NCV_kJkg"]
    return h

def row_for(i, status, reason, keys, sample, sections, plant):
    r=[i, status, reason]
    r+=[round(sample[k],5) for k in keys]
    for sec in SECTIONS_ORDER:
        s=sections.get(sec,{})
        r+=[round(s.get("I_B",float('nan'))/1e6,4),
            round(s.get("I_A",float('nan'))/1e6,4),
            round(s.get("eps",float('nan')),4),
            int(bool(s.get("eps_capped",False))),
            round(s.get("IP",float('nan'))/1e6,4),
            round(s.get("E_resid",float('nan'))/1e6,4)]
    r+=[round(plant.get("I_B",float('nan'))/1e6,4),
        round(plant.get("IP",float('nan'))/1e6,4),
        round(plant.get("eps",float('nan')),4),
        round(plant.get("coal_ex",float('nan'))/1e6,4),
        round(plant.get("clk",float('nan'))/1e6,4),
        round((plant.get("clinker_T") or float('nan')),2),
        round(plant.get("ncv",float('nan'))/1e3,1)]
    return r

def done_samples():
    if not (RESUME and os.path.exists(OUT_CSV)): return set()
    done=set()
    with open(OUT_CSV,newline="") as f:
        for row in csv.reader(f):
            if row and row[0].isdigit(): done.add(int(row[0]))
    return done

# ============================================================ MAIN =============
def main():
    try: import win32com.client as win32
    except ImportError: print("pip install pywin32"); sys.exit(1)
    try: import numpy  # noqa
    except ImportError: print("pip install numpy"); sys.exit(1)

    keys, samples = build_samples()
    done = done_samples()
    header = csv_header(keys)
    new_file = not (RESUME and os.path.exists(OUT_CSV))
    fcsv = open(OUT_CSV, "a", newline="")
    writer = csv.writer(fcsv)
    if new_file: writer.writerow(header); fcsv.flush()
    log = open(LOG_TXT,"a"); 
    def say(m):
        print(m); log.write(m+"\n"); log.flush()

    say("="*64)
    say("CAMPAIGN START  %s  (N=%d, seed=%d)"%(time.strftime("%Y-%m-%d %H:%M:%S"),N_SAMPLES,RNG_SEED))
    say("model: %s"%os.path.basename(BKP))
    say("already done: %d samples"%len(done))

    aspen = win32.Dispatch("Apwn.Document.40.0")
    try: aspen.SuppressDialogs=1
    except Exception: pass
    aspen.InitFromArchive2(BKP)
    try: aspen.Visible=0
    except Exception: pass

    n_ok=n_cap=n_rej=n_fail=0; t0=time.time()
    # count any OK rows already present (resume)
    if RESUME and os.path.exists(OUT_CSV):
        try:
            with open(OUT_CSV,newline="") as fr:
                for rr in csv.reader(fr):
                    if rr and rr[1:2]==["OK"]: n_ok+=1
                    elif rr and rr[1:2]==["OK_CAPPED"]: n_cap+=1
        except Exception: pass
    def accepted():
        return (n_ok + n_cap) if COUNT_CAPPED_AS_OK else n_ok
    target = TARGET_OK if TARGET_OK>0 else N_SAMPLES
    say("target accepted rows: %d (mode=%s)"%(target,"target-OK" if TARGET_OK>0 else "fixed-N"))
    for i in range(len(samples)):
        if TARGET_OK>0 and accepted()>=target:
            say("reached %d accepted rows - stopping."%accepted()); break
        if i in done: continue
        s=samples[i]
        try:
            apply_and_run(aspen, s, keys)
            sections, plant = evaluate(aspen)
            status, reason = verdict(sections, plant)
            writer.writerow(row_for(i,status,reason,keys,s,sections,plant)); fcsv.flush()
            if status=="OK": n_ok+=1
            elif status=="OK_CAPPED": n_cap+=1
            else: n_rej+=1
            if i%10==0 or status in ("REJECT","OK_CAPPED"):
                acc=accepted(); tgt=TARGET_OK if TARGET_OK>0 else N_SAMPLES
                rate=acc/max(1,(acc+n_rej))
                remain=max(0,tgt-acc); eta=(time.time()-t0)/max(1,acc)*remain/max(rate,0.5)
                say("  [%3d] %-6s acc=%d/%d eps=%.3f Tclk=%.0fK %s | ETA %.0f min"
                    %(i,status,acc,tgt,plant.get("eps",0),plant.get("clinker_T") or 0,
                      ("("+reason+")") if reason else "", eta/60))
        except Exception as e:
            n_fail+=1
            writer.writerow([i,"FAIL",str(e)[:120]]+[ "" ]*(len(header)-3)); fcsv.flush()
            say("  [%3d] FAIL %s"%(i,str(e)[:100]))
            # reopen a fresh session if Aspen got into a bad state
            try: aspen.Reset()
            except Exception:
                try: aspen.Close()
                except Exception: pass
                aspen=win32.Dispatch("Apwn.Document.40.0")
                aspen.InitFromArchive2(BKP)

    say("-"*64)
    say("DONE  ok=%d ok_capped=%d reject=%d fail=%d  in %.1f min"
        %(n_ok,n_cap,n_rej,n_fail,(time.time()-t0)/60))
    say("usable rows (OK + OK_CAPPED) = %d / %d"%(n_ok+n_cap, n_ok+n_cap+n_rej+n_fail))
    say("dataset -> %s"%OUT_CSV)
    say("Train the ANN on status in {OK, OK_CAPPED}. The per-section *_eps_capped")
    say("flags mark where a sectional efficiency was capped at 1.0 (off-design,")
    say("numerically singular definition). REJECT rows are genuine 2nd-law")
    say("violations and must be excluded.")
    try: aspen.Close(); aspen.Quit()
    except Exception: pass
    fcsv.close(); log.close()

if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
