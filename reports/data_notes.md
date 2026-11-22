
## Calibration order (2026-11-11)
- The ASLLRP server (dai.cs.rutgers.edu: DAI 2, Sign Bank, downloads) returned "Service Unavailable" from 2026-11-10 evening.
  NCSLGR is only available through the same server (or legacy CD-ROM), so no alternative calibration source exists.
- Deviation from plan 7.1: pass-1 annotation of TRAIN clips started before calibration, to avoid losing annotation time.
  Test and val are annotated only after calibration and pruning. Pruned glosses are removed from train labels
  automatically by build_ctc_files.py. Calibration precision may be slightly optimistic because of annotator practice.
- If the server is still down on 2026-11-13: plan fallback C (no calibration; accuracy reported as unmeasured).
