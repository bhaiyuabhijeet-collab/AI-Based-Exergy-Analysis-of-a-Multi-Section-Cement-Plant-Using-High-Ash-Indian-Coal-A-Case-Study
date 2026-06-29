#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BASECASE_PREFLIGHT  -  GO / NO-GO check before the long campaign
================================================================================
Run BEFORE exergy_campaign_driver.py (the 8-10 h run). The whole pipeline now
lives on the WRITE-THEN-SOLVE branch (Reinit -> write the 26 base inputs ->
Run2 xN), so the base case is ~110.5 MW (not the untouched 108.8 MW branch).
This preflight therefore solves the base the SAME write-then-solve way the
campaign, the calculator, and PSO all use, and checks it against the
write-then-solve base reference. It exercises everything that, if broken,
silently wastes the whole campaign:

  1) CM-GRIND grinding motor reads its real duty (~0.600 MW, not 0)  -> calc fix live
  2) the write-then-solve path converges to the PHYSICAL branch       -> driver path OK
     (RMINLETG ~436 K, plant ~110.5 MW), not the 709 K / ~128 MW branch
  3) the branch is REPRODUCIBLE (two write-then-solve passes agree)    -> single base
  4) per-section I_B, plant total, efficiency, clinker, clinker mass,
     and specific irreversibility match the write-then-solve reference

If everything is green, the pipeline is single-branch and consistent.

RUN (from the AspenPlus folder):
    cd /d "D:\\...\\AspenPlus"
    python basecase_preflight.py
Requires exergy_calculator.py and exergy_campaign_driver.py on the path + .bkp.
================================================================================
"""
import os, sys

# --- known-good base reference: WRITE-THEN-SOLVE branch -----------------------
# Per-section irreversibility I_B [MW] on the write-then-solve branch.
EXPECT_I = {
    "Raw Mill":       7.51,
    "Coal Mill":      1.51,
    "Preheater":     11.43,
    "Calciner":      36.63,
    "Kiln":          37.71,
    "Clinker Cooler":15.72,
}
EXPECT_PLANT_MW   = 110.51      # summed-section Route-B destruction (write-then-solve)
EXPECT_EFF        = 0.3774      # overall plant efficiency (fixed fuel-exergy basis)
EXPECT_CLINKER_MW = 64.45       # clinker product exergy
EXPECT_CMGRIND_MW = 0.600       # grinding-motor duty (CM-GRIND)
EXPECT_SPEC_MJKG  = 1.6403      # specific irreversibility, per-mass (MJ/kg clinker)
EXPECT_CLK_MASS   = 67.37       # clinker mass flow [kg/s]
EXPECT_RMINLETG_K = 436.0       # drying-gas-inlet T on the physical branch (sentinel)

# tolerances
TOL_I_MW     = 0.40             # per-section I_B
TOL_PLANT_MW = 1.50             # plant total
TOL_EFF      = 0.006
TOL_CLK_MW   = 0.80
TOL_GRIND    = 0.10             # +-10% on motor duty
TOL_SPEC     = 0.05             # MJ/kg clinker
TOL_CLK_MASS = 1.00             # kg/s
TOL_REPRO_MW = 0.15             # max plant-I_B spread across two write-then-solve passes
BRANCH_RMG_MAX_K = 600.0        # RMINLETG above this => 709 K (wrong) branch
PHYS_BAND    = (100.0, 120.0)   # coarse physical-branch guard for plant I_B

# ============================================================ IMPORTS ==========
try:
    import exergy_calculator as EX
    import exergy_campaign_driver as DRV
except Exception as e:
    sys.exit("Import failed (run from the AspenPlus folder): %s" % e)
try:
    import win32com.client as win32
except ImportError:
    sys.exit("pip install pywin32")

MW   = 1e6
BKP  = getattr(DRV, "BKP", None)
SETTLE_RUNS = getattr(DRV, "SETTLE_RUNS", 2)
if not BKP or not os.path.exists(BKP):
    sys.exit("model .bkp not found via DRV.BKP: %s" % BKP)

def open_model():
    try:
        asp = win32.Dispatch("Apwn.Document.40.0")
    except Exception:
        asp = win32.Dispatch("Apwn.Document")
    try: asp.SuppressDialogs = 1
    except Exception: pass
    asp.InitFromArchive2(os.path.abspath(BKP))
    try: asp.Visible = 0
    except Exception: pass
    return asp

def ok(flag): return "PASS" if flag else "**FAIL**"

# ============================================================ MAIN =============
def main():
    print("="*70)
    print("BASECASE PRE-FLIGHT  (WRITE-THEN-SOLVE branch, the pipeline's actual path)")
    print("model:", os.path.basename(BKP))
    print("="*70)
    fails = 0
    base = DRV.base_sample()
    keys = list(DRV.INPUTS.keys())
    asp  = open_model()
    try:
        # ---- 1) PRIMARY: write-then-solve the base (Reinit->write->Run2 xN) ----
        DRV.apply_and_run(asp, base, keys)

        # direct motor-duty probe (calculator fix live?)
        try:
            grind = EX.block_duty(asp, "CM-GRIND")/MW
        except Exception:
            grind = float("nan")
        g_ok = abs(grind - EXPECT_CMGRIND_MW) <= TOL_GRIND
        fails += (not g_ok)
        print("\nCM-GRIND duty : %6.3f MW (expect %.3f)   %s"
              % (grind, EXPECT_CMGRIND_MW, ok(g_ok)))
        if not g_ok:
            print("   -> if 0.000, the CM-GRIND DUTY_NODE_OVERRIDE fix is NOT in the calculator on disk.")

        sections, plant = DRV.evaluate(asp)

        # ---- per-section I_B table -----------------------------------------
        print("\n%-15s %9s %9s %8s   %s" % ("Section","I_B[MW]","expect","d","check"))
        print("-"*55)
        for sec, exp in EXPECT_I.items():
            got = sections.get(sec, {}).get("I_B", float("nan"))/MW
            d = got - exp
            f = abs(d) <= TOL_I_MW
            fails += (not f)
            eps = sections.get(sec, {}).get("eps", float("nan"))
            ipv = sections.get(sec, {}).get("IP", float("nan"))/MW
            print("%-15s %9.3f %9.3f %+8.3f   %s   (eps=%.1f%%, IP=%.2f)"
                  % (sec, got, exp, d, ok(f),
                     (eps*100 if eps==eps else float("nan")), ipv))

        # ---- plant aggregates ----------------------------------------------
        ptot = plant.get("I_B", float("nan"))/MW
        eff  = plant.get("eps", float("nan"))
        clk  = plant.get("clk", float("nan"))/MW
        p_ok = abs(ptot - EXPECT_PLANT_MW) <= TOL_PLANT_MW
        e_ok = abs(eff  - EXPECT_EFF)       <= TOL_EFF
        c_ok = abs(clk  - EXPECT_CLINKER_MW)<= TOL_CLK_MW
        fails += (not p_ok) + (not e_ok) + (not c_ok)
        print("-"*55)
        print("plant total   : %8.3f MW (expect %.2f)   %s" % (ptot, EXPECT_PLANT_MW, ok(p_ok)))
        print("efficiency    : %8.3f    (expect %.3f)   %s  (fixed fuel-exergy basis)" % (eff, EXPECT_EFF, ok(e_ok)))
        print("clinker exergy: %8.3f MW (expect %.2f)   %s" % (clk,  EXPECT_CLINKER_MW, ok(c_ok)))

        # ---- specific irreversibility (per-mass, MJ/kg clinker) -------------
        m_clk = plant.get("m_clinker", float("nan"))
        spec  = plant.get("spec_IB_MJkg", float("nan"))
        if not (spec == spec) and (m_clk == m_clk) and m_clk > 0:   # NaN-safe fallback
            spec = (plant.get("I_B", float("nan")))/m_clk/MW        # W/(kg/s)/1e6 = MJ/kg
        s_ok = (spec == spec) and abs(spec - EXPECT_SPEC_MJKG) <= TOL_SPEC
        m_ok = (m_clk == m_clk) and abs(m_clk - EXPECT_CLK_MASS) <= TOL_CLK_MASS
        fails += (not s_ok) + (not m_ok)
        print("clinker mass  : %8.3f kg/s (expect %.2f)   %s" % (m_clk, EXPECT_CLK_MASS, ok(m_ok)))
        print("specific irrev: %8.4f MJ/kg (expect %.4f)   %s" % (spec, EXPECT_SPEC_MJKG, ok(s_ok)))

        # ---- 2) PHYSICAL-BRANCH SENTINEL (RMINLETG temperature) -------------
        # On the correct (physical) branch RMINLETG ~436 K; the spurious second
        # steady state runs the drying-gas loop hot (~709 K) and inflates plant I.
        rmg = plant.get("rminletg_T", float("nan"))
        band_ok = (ptot == ptot) and (PHYS_BAND[0] <= ptot <= PHYS_BAND[1])
        rmg_ok  = (rmg == rmg) and (rmg < BRANCH_RMG_MAX_K)
        fails += (not band_ok) + (not rmg_ok)
        print("\n[branch sentinel]")
        print("   plant I_B in physical band %.0f-%.0f MW : %8.3f   %s"
              % (PHYS_BAND[0], PHYS_BAND[1], ptot, ok(band_ok)))
        print("   RMINLETG T (expect ~%.0f K, <%.0f)       : %8.2f   %s%s"
              % (EXPECT_RMINLETG_K, BRANCH_RMG_MAX_K, rmg, ok(rmg_ok),
                 "   (709 K branch!)" if not rmg_ok else ""))

        # ---- info: Route A vs Route B (no pass/fail) ------------------------
        rA = plant.get("I_A", float("nan"))/MW
        if rA == rA and ptot == ptot:
            print("   Route A (T0*Sgen) = %.3f MW vs Route B = %.3f MW  (gap %+.1f%%, informational)"
                  % (rA, ptot, 100.0*(rA-ptot)/ptot if ptot else float("nan")))

        # ---- 3) BRANCH REPRODUCIBILITY: solve write-then-solve AGAIN --------
        # A second independent write-then-solve must land on the SAME value;
        # this is what guarantees a single, stable base across the pipeline.
        print("\n[reproducibility] second write-then-solve pass should match the first:")
        try:
            DRV.apply_and_run(asp, base, keys)
            s2, p2 = DRV.evaluate(asp)
            ptot2 = p2.get("I_B", float("nan"))/MW
            spread = abs(ptot2 - ptot)
            r_ok = (ptot2 == ptot2) and spread <= TOL_REPRO_MW
            fails += (not r_ok)
            print("   pass#1 = %.4f MW   pass#2 = %.4f MW   |spread| = %.4f MW   %s"
                  % (ptot, ptot2, spread, ok(r_ok)))
            if not r_ok:
                print("   -> branch not reproducible; the base value is path-unstable.")
        except Exception as e:
            fails += 1
            print("   second pass FAILED: %s   **FAIL**" % str(e)[:90])
    finally:
        try: asp.Close(); asp.Quit()
        except Exception: pass

    print("\n" + "="*70)
    if fails == 0:
        print("RESULT:  GO  -  base reproduces the write-then-solve reference (single branch).")
        print("         Pipeline is consistent; safe to launch the 8-10 h campaign.")
    else:
        print("RESULT:  NO-GO  -  %d check(s) failed above." % fails)
        print("         Fix these before launching, or the long campaign will be wrong.")
    print("="*70)
    sys.exit(0 if fails == 0 else 1)

if __name__ == "__main__":
    main()
