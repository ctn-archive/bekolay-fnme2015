#!/usr/bin/env bash
export COLUMNS=50
export PYTEST_ADDOPTS="-p nengo.tests.options"

for py in $( ls test*.py ); do
    out="$( basename $py .py ).out"
    py.test -q --plots --analytics --logs -- $py > $out
done

types=(plots analytics logs)
for typ in "${types[@]}"; do
    tree "nengo.simulator.$typ" > "$typ.out"
done

pdf2svg "nengo.simulator.plots/LIF/test_ensemble.pdf" "../img/plt.svg"
