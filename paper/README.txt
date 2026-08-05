SonoSPADE — MICCAI 2026 SASHIMI submission (Overleaf package)
=============================================================

Open in Overleaf
----------------
1. Overleaf -> New Project -> Upload Project -> select this .zip
   (upload the ZIP itself, so the figures/ subfolder is preserved).
2. Menu -> Settings -> Compiler: pdfLaTeX;  TeX Live: latest.
3. Menu -> Settings -> Main document: main_sashimi.tex
4. Recompile. It runs pdfLaTeX -> BibTeX -> pdfLaTeX x2.

The main paper is 8 pages + references. The supplementary is a
separate document in the same project: to build it, switch
"Main document" to supplementary.tex and recompile.

Files
-----
main_sashimi.tex       Main paper (8 pages + references), Springer LLNCS class.
supplementary.tex      Supplementary material: Tables S1-S2 and Fig. S1.
results_macros.tex     Reported numbers (\input by both documents).
rl_results_macros.tex  In-the-loop numbers (\input by both documents).
references.bib         Bibliography, splncs04 style.
main_sashimi.bbl       Precompiled bibliography (builds even without a BibTeX pass).
llncs.cls              Springer LLNCS class (Overleaf also provides this).
splncs04.bst           Springer LLNCS bib style.
figures/               qualitative.png, curvilinear.png.

Before you submit
-----------------
- Add the author name(s), affiliations, and email on the title block of
  main_sashimi.tex (currently the anonymous block). Keep it anonymous if the
  workshop review is double-blind.
- Confirm the SASHIMI page limit is 8 pages excluding references.
