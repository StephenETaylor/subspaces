# intended to be invoked (e.g.):  gnuplot -e "name='f'" g9b.gnuplot
# creates a PDF file with a number of surface plots in it.
# page 5 and page 9 are particularly interesting.

outfile = name.".pdf"
infile = name.".gp"
set term pdfcairo
set output outfile

set xlabel 'pairs heldout'
set ylabel 'words pinned'
set zlabel 'accuracy' rotate parallel

# calculate number of untrained and trained pairs, calculations isolated here
ut(x) = x #use only as ut($2)
tr(x,y) = 1+int(sqrt(x)) - ut(y) # use only as tr($6,$2)
xm(x) = x*(x-1) # to combine tr-tr or ut-ut, need to eliminate diagonal 

splot infile using ($2):($4):(($8+$16+$17+$18+$19 - $21 -$22 - $23 -$24 )/$6) title 'performance after training'

splot infile using ($2):($4):(($11+$16- $21 )/xm(tr($6,$2)))
splot infile using ($2):($4):(($12+$17- $22 )/ut($2)/tr($6,$2)) title 'trained::untrained'
splot infile using ($2):($4):(($13+$18- $23 )/ut($2)/tr($6,$2)) title 'untrained::trained'

splot infile using ($2):($4):(($14+$19- $24 )/xm(ut($2)))  with surface title 'untrained::untrained'

splot infile using ($2):($4):(($16- $21 )/xm(tr($6,$2))) title 'imp. trained::trained'
splot infile using ($2):($4):(($17- $22 )/ut($2)/tr($6,$2)) title 'imp. trained::untrained'
splot infile using ($2):($4):(($18- $23 )/ut($2)/tr($6,$2)) title 'imp. untrained::trained'
splot infile using ($2):($4):(($19- $24 )/xm(ut($2)))  title 'imp. untrained::untrained'
