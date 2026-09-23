#!/bin/bash
# Usage: _render_test.sh <fig_basename_no_ext> <full|col>
# Renders every page of the test PDF, then reports the largest-file-size
# page as OUTPUT (that's the one that actually contains the figure --
# figure*[t] in twocolumn often floats past the title/abstract page).
set -e
FIG="$1"
MODE="$2"
cd "$(dirname "$0")"
if [ "$MODE" = "col" ]; then
  TEMPLATE="_test_page_col.tex"
else
  TEMPLATE="_test_page.tex"
fi
rm -f "_test_${FIG}"*.png "_test_${FIG}.pdf" "_test_${FIG}.log"
sed "s/FIGNAME/${FIG}/" "$TEMPLATE" > "_test_${FIG}.tex"
pdflatex -interaction=nonstopmode -halt-on-error "_test_${FIG}.tex" > "_test_${FIG}.log" 2>&1 || (tail -40 "_test_${FIG}.log" && exit 1)
pdftoppm -png -r 150 "_test_${FIG}.pdf" "_test_${FIG}"
BEST=$(ls -S "_test_${FIG}"-*.png | head -1)
cp "$BEST" "_test_${FIG}_OUTPUT.png"
echo "Picked page: $BEST"
ls -la "_test_${FIG}"*.png
