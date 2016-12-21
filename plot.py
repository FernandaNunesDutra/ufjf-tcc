# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

N = 8
menMeans = (58,42,46,37,35,32,32,43)
#menMeans = (18,24,28,19,26,22,25,23,26)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='#1a2fed')

womenMeans = (0,5,1,3,1,1,3,0)
#womenMeans = (0,0,0,0,0,0,1,0,0)
rects2 = ax.bar(ind + width, womenMeans, width, color='#ef7f7f')

# add some text for labels, title and axes ticks
ax.set_ylabel('avaliacoes')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('item 1', 'item 2', 'item 2', 'item 4', 'item 5', 'item 6', 'item 7', 'item 8', 'item 9'))

fontP = FontProperties()
fontP.set_size('small')
ax.legend((rects1[0], rects2[0]), ('positivas', 'negativas'), prop = fontP)

plt.savefig('aes-plot.eps', format='eps', dpi=1000)
#plt.show()
