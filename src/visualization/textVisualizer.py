import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import seaborn as sn
import sys
import re
import csv
from itertools import chain


def visualizeData(file, compfile, source, class_):

	if(source == 'reddit' or source == 'combined'):
		labels = ['Spelling Delta', 'Caps', 'Text length', 'Word length', 'Punctuation', 'Adjective', 'Interjection', 'Common Noun', 'Proper Noun', 'Verb', 'Pronoun', 'Adverb']
		features = ['spellDelta', 'caps', 'textLength', 'sentenceWordLength', ',', 'A', '!', 'N', '^', 'V', 'O', 'R',  class_]
	else:
		labels = ['Elongation', 'Caps', 'Text length', 'Word length', 'Punctuation', 'Adjective', 'Interjection', 'Common Noun', 'Proper Noun', 'Verb', 'Pronoun', 'Adverb']
		features = ['elongated', 'caps', 'textLength', 'sentenceWordLength', ',', 'A', '!', 'N', '^', 'V', 'O', 'R',  class_]



	data = pd.read_csv(file, header=0, sep=',')
	comparison = pd.read_csv(compfile, header=0, sep=',')

	featuredata = data[features]
	comparisondata = comparison[features]





	

	# ax1.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[0]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[0]]):
	#     ax1.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax1.set_title(labels[0], fontdict={'size':12})
	# ax1.set(ylabel='Average amount per text')
	# ax1.set_ylim(0, 0.18)
	# ax1.axis('square') 
	# #ax1.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[1]])
	# ax2.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[1]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[1]]):
	#     ax2.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax2.set_title(labels[1], fontdict={'size':12})
	# ax2.set_ylim(0, 0.8)
	# ax2.axis('scaled') 
	# #ax2.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[2]])
	# ax3.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[2]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[2]]):
	#     ax3.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax3.set_title(labels[2], fontdict={'size':12})
	# ax3.set_ylim(0, 18)
	# #ax3.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[3]])
	# ax4.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[3]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[3]]):
	#     ax4.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax4.set_title(labels[3], fontdict={'size':12})
	# ax4.set_ylim(0, 5)
	# #ax4.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[4]])
	# ax5.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[4]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[4]]):
	#     ax5.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax5.set_title(labels[4], fontdict={'size':12})
	# ax5.set(ylabel='Percentage of words in text')
	# ax5.set_ylim(0, 55)
	# #ax5.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[5]])
	# ax6.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[5]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[5]]):
	#     ax6.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax6.set_title(labels[5], fontdict={'size':12})
	# ax6.set_ylim(0, 15)
	# #ax6.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[6]])
	# ax7.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[6]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[6]]):
	#     ax7.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax7.set_title(labels[6], fontdict={'size':12})
	# ax7.set_ylim(0, 8)
	# #ax7.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[7]])
	# ax8.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[7]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[7]]):
	#     ax8.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax8.set_title(labels[7], fontdict={'size':12})
	# ax8.set_ylim(0, 65)
	# #ax8.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[8]])
	# ax9.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[8]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[8]]):
	#     ax9.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax9.set_title(labels[8], fontdict={'size':12})
	# ax9.set(ylabel='Percentage of words in text')
	# ax9.set_ylim(0,25)

	# plt.xticks(featuredata.index, featuredata[class_], rotation=50, horizontalalignment='center', fontsize=6)
	# #ax9.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[9]])
	# ax10.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[9]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[9]]):
	#     ax10.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax10.set_title(labels[9], fontdict={'size':12})

	# #plt.xticks(featuredata.index, featuredata[class_], rotation=50, horizontalalignment='center', fontsize=6)
	# ax10.set_ylim(0, 37)
	# #ax10.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[10]])
	# ax11.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[10]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[10]]):
	#     ax11.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax11.set_title(labels[10], fontdict={'size':12})

	# #plt.xticks(featuredata.index, featuredata[class_], rotation=50, horizontalalignment='center', fontsize=6)
	# ax11.set_ylim(0, 15)
	# #ax11.set_xticks(featuredata.index, featuredata[class_], rotation=60, horizontalalignment='right', fontsize=12)

	# print(featuredata[features[11]])
	# ax12.vlines(x=featuredata[class_], ymin=0, ymax=featuredata[features[11]], color='firebrick', alpha=0.7, linewidth=10)
	# for i, cty in enumerate(featuredata[features[11]]):
	#     ax12.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
	# ax12.set_title(labels[11], fontdict={'size':12})
	# ax12.set_ylim(0, 9);


	# plt.xticks(featuredata.index, featuredata[class_], rotation=50, horizontalalignment='center', fontsize=6)
	#ax1.errorbar(featuredata[features[0]], data[class_], xerr=stddata[features[0]],fmt='+', solid_capstyle='projecting', capsize=5)

	if(class_ == 'lang'):
		if(source == 'twitter'):
			scale = 0.1

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(10,7))

			print(featuredata[features[0]])
			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax1.xaxis.set_ticks(np.arange(0, featuredata[features[0]].max() + 0.1, 0.1))
			ax1.set_xlim(0, 0.22);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])


			

			print(featuredata[features[1]])
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax2.xaxis.set_ticks(np.arange(0, featuredata[features[1]].max() + 0.2, 0.1))
			ax2.set_xlim(0.20, 0.72);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]])
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax3.xaxis.set_ticks(np.arange(0, featuredata[features[2]].max() + 10, 2))
			ax3.set_xlim(4.4, 18.5);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]])
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.2))
			ax4.set_xlim(3.8, 4.92);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]])
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax5.xaxis.set_ticks(np.arange(0, featuredata[features[4]].max() + 4, 6))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(13, 55);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]])
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax6.xaxis.set_ticks(np.arange(0, featuredata[features[5]].max() + 1, 2))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax6.set_xlim(6, 15);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]])
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax7.xaxis.set_ticks(np.arange(0, featuredata[features[6]].max() + 0.5, 1))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax7.set_xlim(0, 8);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]])
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax8.xaxis.set_ticks(np.arange(0, featuredata[features[7]].max() + 3, 6))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax8.set_xlim(25, 66);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]])
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax9.xaxis.set_ticks(np.arange(0, featuredata[features[8]].max() + 4, 3))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(7, 25);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]])
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax10.xaxis.set_ticks(np.arange(0, featuredata[features[9]].max() + 3, 5))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax10.set_xlim(14, 36.5);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]])
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax11.xaxis.set_ticks(np.arange(0, featuredata[features[10]].max() + 6, 3))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax11.set_xlim(0, 15);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]])
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax12.xaxis.set_ticks(np.arange(0, featuredata[features[11]].max() + 1, 1))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax12.set_xlim(2, 8);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])

			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				#ax.tick_params(axis='y', labelsize=7)
		elif(source == 'reddit'):
			scale = 0.1

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(10,10))

			print(featuredata[features[0]])
			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='European', markersize=3)
			ax1.plot(comparisondata[features[0]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			#ax1.hlines(y=data[class_], xmin=0, xmax=featuredata[features[0]], color='firebrick', alpha=0.7, linewidth=10)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[0]]):
				if(y == comparisondata[features[0]].max() or y == comparisondata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax1.xaxis.set_ticks(np.arange(0, featuredata[features[0]].max() + 0.1, 0.2))
			ax1.set_xlim(1.6, 2.85);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])
			#ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), prop={'size': 6})
			ax1.legend(loc='lower center', bbox_to_anchor=(-0.6,0.9), prop={'size': 6})


			

			print(featuredata[features[1]])
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='European', markersize=3)
			ax2.plot(comparisondata[features[1]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[1]]):
				if(y == comparisondata[features[1]].max() or y == comparisondata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax2.xaxis.set_ticks(np.arange(0, featuredata[features[1]].max() + 0.2, 0.3))
			ax2.set_xlim(0.08, 1.8);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]])
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='European', markersize=3)
			ax3.plot(comparisondata[features[2]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[2]]):
				if(y == comparisondata[features[2]].max() or y == comparisondata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax3.xaxis.set_ticks(np.arange(0, featuredata[features[2]].max() + 10, 6))
			ax3.set_xlim(11, 70);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]])
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='European', markersize=3)
			ax4.plot(comparisondata[features[3]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[3]]):
				if(y == comparisondata[features[3]].max() or y == comparisondata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.2))
			ax4.set_xlim(3.9, 5.1);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]])
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='European', markersize=3)
			ax5.plot(comparisondata[features[4]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					if(x == 0):
						ax5.text(y+0.3, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[4]]):
				if(y == comparisondata[features[4]].max() or y == comparisondata[features[4]].min()):
					if(x == 0):
						ax5.text(y-0.3, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax5.xaxis.set_ticks(np.arange(0, featuredata[features[4]].max() + 4, 3))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(15.7, 32);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]])
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='European', markersize=3)
			ax6.plot(comparisondata[features[5]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[5]]):
				if(y == comparisondata[features[5]].max() or y == comparisondata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax6.xaxis.set_ticks(np.arange(0, featuredata[features[5]].max() + 1, 1))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax6.set_xlim(4.5, 8.8);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]])
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='European', markersize=3)
			ax7.plot(comparisondata[features[6]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[6]]):
				if(y == comparisondata[features[6]].max() or y == comparisondata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax7.xaxis.set_ticks(np.arange(0, featuredata[features[6]].max() + 0.5, 0.5))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax7.set_xlim(1.9, 5);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]])
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='European', markersize=3)
			ax8.plot(comparisondata[features[7]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[7]]):
				if(y == comparisondata[features[7]].max() or y == comparisondata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax8.xaxis.set_ticks(np.arange(0, featuredata[features[7]].max() + 3, 1))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax8.set_xlim(16, 22.5);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]])
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='European', markersize=3)
			ax9.plot(comparisondata[features[8]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[8]]):
				if(y == comparisondata[features[8]].max() or y == comparisondata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax9.xaxis.set_ticks(np.arange(0, featuredata[features[8]].max() + 4, 2))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(15, 26.5);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]])
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='European', markersize=3)
			ax10.plot(comparisondata[features[9]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[9]]):
				if(y == comparisondata[features[9]].max() or y == comparisondata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax10.xaxis.set_ticks(np.arange(0, featuredata[features[9]].max() + 3, 1))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax10.set_xlim(10, 17);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]])
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='European', markersize=3)
			ax11.plot(comparisondata[features[10]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[10]]):
				if(y == comparisondata[features[10]].max() or y == comparisondata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax11.xaxis.set_ticks(np.arange(0, featuredata[features[10]].max() + 6, 1))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax11.set_xlim(2, 8);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]])
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='European', markersize=3)
			ax12.plot(comparisondata[features[11]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[11]]):
				if(y == comparisondata[features[11]].max() or y == comparisondata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax12.xaxis.set_ticks(np.arange(0, featuredata[features[11]].max() + 1, 0.5))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax12.set_xlim(2.5, 4.1);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])

			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				ax.tick_params(axis='y', labelsize=7)
		else:
			scale = 0.1

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(10,10))

			print(featuredata[features[0]].min(), featuredata[features[0]].max(), comparisondata[features[0]].min(), comparisondata[features[0]].max())
			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='European', markersize=3)
			ax1.plot(comparisondata[features[0]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			#ax1.hlines(y=data[class_], xmin=0, xmax=featuredata[features[0]], color='firebrick', alpha=0.7, linewidth=10)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[0]]):
				if(y == comparisondata[features[0]].max() or y == comparisondata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax1.xaxis.set_ticks(np.arange(0, 2.8, 0.2))
			ax1.set_xlim(1.15, 2.7);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])
			#ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), prop={'size': 6})
			ax1.legend(loc='lower center', bbox_to_anchor=(-0.6,0.9), prop={'size': 6})


			

			print(featuredata[features[1]].min(), featuredata[features[1]].max(), comparisondata[features[1]].min(), comparisondata[features[1]].max())
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='European', markersize=3)
			ax2.plot(comparisondata[features[1]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[1]]):
				if(y == comparisondata[features[1]].max() or y == comparisondata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax2.xaxis.set_ticks(np.arange(0, 1.8, 0.3))
			ax2.set_xlim(0.08, 1.7);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]].min(), featuredata[features[2]].max(), comparisondata[features[2]].min(), comparisondata[features[2]].max())
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='European', markersize=3)
			ax3.plot(comparisondata[features[2]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[2]]):
				if(y == comparisondata[features[2]].max() or y == comparisondata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax3.xaxis.set_ticks(np.arange(0, 70, 6))
			ax3.set_xlim(2, 70);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]].min(), featuredata[features[3]].max(), comparisondata[features[3]].min(), comparisondata[features[3]].max())
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='European', markersize=3)
			ax4.plot(comparisondata[features[3]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[3]]):
				if(y == comparisondata[features[3]].max() or y == comparisondata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.2))
			ax4.set_xlim(3.9, 5.1);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]].min(), featuredata[features[4]].max(), comparisondata[features[4]].min(), comparisondata[features[4]].max())
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='European', markersize=3)
			ax5.plot(comparisondata[features[4]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					#if(x == 0):
					#	ax5.text(y+0.3, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					#else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[4]]):
				if(y == comparisondata[features[4]].max() or y == comparisondata[features[4]].min()):
					#if(x == 0):
					#	ax5.text(y-0.3, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					#else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax5.xaxis.set_ticks(np.arange(0, 60, 8))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(13, 56);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]].min(), featuredata[features[5]].max(), comparisondata[features[5]].min(), comparisondata[features[5]].max())
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='European', markersize=3)
			ax6.plot(comparisondata[features[5]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[5]]):
				if(y == comparisondata[features[5]].max() or y == comparisondata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax6.xaxis.set_ticks(np.arange(0, 15, 2))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax6.set_xlim(4, 14);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]].min(), featuredata[features[6]].max(), comparisondata[features[6]].min(), comparisondata[features[6]].max())
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='European', markersize=3)
			ax7.plot(comparisondata[features[6]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[6]]):
				if(y == comparisondata[features[6]].max() or y == comparisondata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax7.xaxis.set_ticks(np.arange(0, 10, 1))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax7.set_xlim(0, 8);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]].min(), featuredata[features[7]].max(), comparisondata[features[7]].min(), comparisondata[features[7]].max())
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='European', markersize=3)
			ax8.plot(comparisondata[features[7]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[7]]):
				if(y == comparisondata[features[7]].max() or y == comparisondata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax8.xaxis.set_ticks(np.arange(0, 55, 5))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax8.set_xlim(14, 52);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]].min(), featuredata[features[8]].max(), comparisondata[features[8]].min(), comparisondata[features[8]].max())
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='European', markersize=3)
			ax9.plot(comparisondata[features[8]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[8]]):
				if(y == comparisondata[features[8]].max() or y == comparisondata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax9.xaxis.set_ticks(np.arange(0, 30, 3))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(10, 26.5);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]].min(), featuredata[features[9]].max(), comparisondata[features[9]].min(), comparisondata[features[9]].max())
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='European', markersize=3)
			ax10.plot(comparisondata[features[9]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[9]]):
				if(y == comparisondata[features[9]].max() or y == comparisondata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax10.xaxis.set_ticks(np.arange(0, 40, 4))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax10.set_xlim(8, 32);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]].min(), featuredata[features[10]].max(), comparisondata[features[10]].min(), comparisondata[features[10]].max())
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='European', markersize=3)
			ax11.plot(comparisondata[features[10]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[10]]):
				if(y == comparisondata[features[10]].max() or y == comparisondata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax11.xaxis.set_ticks(np.arange(0, 10, 1))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax11.set_xlim(1.5, 8);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]].min(), featuredata[features[11]].max(), comparisondata[features[11]].min(), comparisondata[features[11]].max())
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='European', markersize=3)
			ax12.plot(comparisondata[features[11]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[11]]):
				if(y == comparisondata[features[11]].max() or y == comparisondata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax12.xaxis.set_ticks(np.arange(0, 7, 1))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax12.set_xlim(2, 6);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])

			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				ax.tick_params(axis='y', labelsize=7)

		
	elif(class_ == 'langFam'):
		if(source == 'twitter'):
			print('obsolete')
		else:
			scale = 0

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(10,6))

			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='European', markersize=3)
			ax1.plot(comparisondata[features[0]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			#ax1.hlines(y=data[class_], xmin=0, xmax=featuredata[features[0]], color='firebrick', alpha=0.7, linewidth=10)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					if(x == 2):
						continue
					else:
						ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[0]]):
				if(y == comparisondata[features[0]].max() or y == comparisondata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax1.xaxis.set_ticks(np.arange(0, featuredata[features[0]].max() + 0.1, 0.1))
			ax1.set_xlim(1.85, 2.3);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])
			#ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), prop={'size': 6})
			ax1.legend(loc='lower center', bbox_to_anchor=(-0.6,0.9), prop={'size': 6})

			

			print(featuredata[features[1]])
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='European', markersize=3)
			ax2.plot(comparisondata[features[1]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)

			for x, y in enumerate(comparisondata[features[1]]):
				if(y == comparisondata[features[1]].max() or y == comparisondata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)

			ax2.xaxis.set_ticks(np.arange(0, featuredata[features[1]].max() + 0.2, 0.1))
			ax2.set_xlim(0.28, 0.62);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]])
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='European', markersize=3)
			ax3.plot(comparisondata[features[2]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					if(x == 2):
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[2]]):
				if(y == comparisondata[features[2]].max() or y == comparisondata[features[2]].min()):
					if(x == 2):
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax3.xaxis.set_ticks(np.arange(0, featuredata[features[2]].max() + 10, 4))
			ax3.set_xlim(19.5, 38);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]])
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='European', markersize=3)
			ax4.plot(comparisondata[features[3]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[3]]):
				if(y == comparisondata[features[3]].max() or y == comparisondata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.1))
			ax4.set_xlim(4.25, 4.82);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]])
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='European', markersize=3)
			ax5.plot(comparisondata[features[4]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					if(x == 2):
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[4]]):
				if(y == comparisondata[features[4]].max() or y == comparisondata[features[4]].min()):
					if(x == 2):
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax5.xaxis.set_ticks(np.arange(0, featuredata[features[4]].max() + 4, 1))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(20.5, 26.5);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]])
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='European', markersize=3)
			ax6.plot(comparisondata[features[5]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[5]]):
				if(y == comparisondata[features[5]].max() or y == comparisondata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax6.xaxis.set_ticks(np.arange(0, featuredata[features[5]].max() + 1, 0.5))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax6.set_xlim(6, 8.6);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]])
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='European', markersize=3)
			ax7.plot(comparisondata[features[6]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[6]]):
				if(y == comparisondata[features[6]].max() or y == comparisondata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax7.xaxis.set_ticks(np.arange(0, featuredata[features[6]].max() + 0.5, 0.2))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax7.set_xlim(3.2, 4.1);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]])
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='European', markersize=3)
			ax8.plot(comparisondata[features[7]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[7]]):
				if(y == comparisondata[features[7]].max() or y == comparisondata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax8.xaxis.set_ticks(np.arange(0, featuredata[features[7]].max() + 3, 0.5))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax8.set_xlim(17.6, 20.3);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]])
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='European', markersize=3)
			ax9.plot(comparisondata[features[8]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[8]]):
				if(y == comparisondata[features[8]].max() or y == comparisondata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax9.xaxis.set_ticks(np.arange(0, featuredata[features[8]].max() + 4, 1))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(19.8, 24);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]])
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='European', markersize=3)
			ax10.plot(comparisondata[features[9]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[9]]):
				if(y == comparisondata[features[9]].max() or y == comparisondata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax10.xaxis.set_ticks(np.arange(0, featuredata[features[9]].max() + 3, 0.5))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax10.set_xlim(11.8, 14);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]])
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='European', markersize=3)
			ax11.plot(comparisondata[features[10]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[10]]):
				if(y == comparisondata[features[10]].max() or y == comparisondata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax11.xaxis.set_ticks(np.arange(0, featuredata[features[10]].max() + 6, 1))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax11.set_xlim(4, 6.5);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]])
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='European', markersize=3)
			ax12.plot(comparisondata[features[11]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[11]]):
				if(y == comparisondata[features[11]].max() or y == comparisondata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax12.xaxis.set_ticks(np.arange(0, featuredata[features[11]].max() + 1, 0.2))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax12.set_xlim(3.1, 3.9);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])


			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				#ax.tick_params(axis='y', labelsize=7)
	elif(class_ == 'origin'):
		if(source == 'twitter'):
			scale = 0

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(8,6))

			print(featuredata[features[0]])
			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax1.xaxis.set_ticks(np.arange(0, featuredata[features[0]].max() + 0.1, 0.02))
			ax1.set_xlim(0.1, 0.18);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])


			

			print(featuredata[features[1]])
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax2.xaxis.set_ticks(np.arange(0, featuredata[features[1]].max() + 0.2, 0.1))
			ax2.set_xlim(0.3, 0.62);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]])
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax3.xaxis.set_ticks(np.arange(0, featuredata[features[2]].max() + 10, 2))
			ax3.set_xlim(11, 15);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]])
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.05))
			ax4.set_xlim(4.55, 4.65);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]])
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax5.xaxis.set_ticks(np.arange(0, featuredata[features[4]].max() + 4, 5))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(15, 33);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]])
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax6.xaxis.set_ticks(np.arange(0, featuredata[features[5]].max() + 1, 2))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax6.set_xlim(6, 11.3);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]])
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax7.xaxis.set_ticks(np.arange(0, featuredata[features[6]].max() + 0.5, 0.5))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax7.set_xlim(2, 3.5);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]])
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax8.xaxis.set_ticks(np.arange(0, featuredata[features[7]].max() + 3, 6))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax8.set_xlim(25, 45);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]])
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax9.xaxis.set_ticks(np.arange(0, featuredata[features[8]].max() + 4, 3))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(7, 16);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]])
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax10.xaxis.set_ticks(np.arange(0, featuredata[features[9]].max() + 3, 5))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax10.set_xlim(14, 25);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]])
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax11.xaxis.set_ticks(np.arange(0, featuredata[features[10]].max() + 6, 1))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax11.set_xlim(3, 6);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]])
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='Twitter', markersize=4)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=6)
			ax12.xaxis.set_ticks(np.arange(0, featuredata[features[11]].max() + 1, 0.5))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax12.set_xlim(3, 4.5);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])

			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				#ax.tick_params(axis='y', labelsize=7)
		else:
			scale = 0.1

			fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(8,4))

			ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='European', markersize=3)
			ax1.plot(comparisondata[features[0]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax1.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[0]]):
				if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
					if(x == 0):
						continue
					else:
						ax1.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
			for x, y in enumerate(comparisondata[features[0]]):
				if(y == comparisondata[features[0]].max() or y == comparisondata[features[0]].min()):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax1.xaxis.set_ticks(np.arange(0, featuredata[features[0]].max() + 0.1, 0.1))
			ax1.set_xlim(1.85, 2.2);
			ax1.set_ylim(-0.5, len(data[class_])+scale)
			ax1.set_title(labels[0])
			ax1.legend(loc='lower center', bbox_to_anchor=(-0.6,0.9), prop={'size': 6})

			

			print(featuredata[features[1]])
			ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='European', markersize=3)
			ax2.plot(comparisondata[features[1]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax2.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[1]]):
				if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)

			for x, y in enumerate(comparisondata[features[1]]):
				if(y == comparisondata[features[1]].max() or y == comparisondata[features[1]].min()):
					ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)

			ax2.xaxis.set_ticks(np.arange(0, featuredata[features[1]].max() + 0.2, 0.1))
			ax2.set_xlim(0.28, 0.53);
			ax2.set_ylim(-0.5, len(data[class_])+scale)
			ax2.set_title(labels[1])

			print(featuredata[features[2]])
			ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='European', markersize=3)
			ax3.plot(comparisondata[features[2]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax3.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[2]]):
				if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
					if(x == 0):
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[2]]):
				if(y == comparisondata[features[2]].max() or y == comparisondata[features[2]].min()):
					if(x == 0):
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax3.xaxis.set_ticks(np.arange(0, featuredata[features[2]].max() + 10, 2))
			ax3.set_xlim(19.5, 30.5);
			ax3.set_ylim(-0.5, len(data[class_])+scale)
			ax3.set_title(labels[2])

			print(featuredata[features[3]])
			ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='European', markersize=3)
			ax4.plot(comparisondata[features[3]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax4.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[3]]):
				if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[3]]):
				if(y == comparisondata[features[3]].max() or y == comparisondata[features[3]].min()):
					ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax4.xaxis.set_ticks(np.arange(0, featuredata[features[3]].max() + 0.2, 0.1))
			ax4.set_xlim(4.25, 4.82);
			ax4.set_ylim(-0.5, len(data[class_])+scale)
			ax4.set_title(labels[3])

			print(featuredata[features[4]])
			ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='European', markersize=3)
			ax5.plot(comparisondata[features[4]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax5.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[4]]):
				if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
					if(x == 0):
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[4]]):
				if(y == comparisondata[features[4]].max() or y == comparisondata[features[4]].min()):
					if(x == 0):
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax5.xaxis.set_ticks(np.arange(0, featuredata[features[4]].max() + 4, 2))
			ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax5.set_xlim(20.5, 26.2);
			ax5.set_ylim(-0.5, len(data[class_])+scale)
			ax5.set_title(labels[4])

			print(featuredata[features[5]])
			ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='European', markersize=3)
			ax6.plot(comparisondata[features[5]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax6.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[5]]):
				if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[5]]):
				if(y == comparisondata[features[5]].max() or y == comparisondata[features[5]].min()):
					ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax6.xaxis.set_ticks(np.arange(0, featuredata[features[5]].max() + 3, 1))
			ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax6.set_xlim(6, 8.6);
			ax6.set_ylim(-0.5, len(data[class_])+scale)
			ax6.set_title(labels[5])

			print(featuredata[features[6]])
			ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='European', markersize=3)
			ax7.plot(comparisondata[features[6]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax7.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[6]]):
				if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[6]]):
				if(y == comparisondata[features[6]].max() or y == comparisondata[features[6]].min()):
					ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax7.xaxis.set_ticks(np.arange(0, featuredata[features[6]].max() + 0.5, 0.2))
			ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax7.set_xlim(3.50, 4.1);
			ax7.set_ylim(-0.5, len(data[class_])+scale)
			ax7.set_title(labels[6])

			print(featuredata[features[7]])
			ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='European', markersize=3)
			ax8.plot(comparisondata[features[7]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax8.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[7]]):
				if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[7]]):
				if(y == comparisondata[features[7]].max() or y == comparisondata[features[7]].min()):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax8.xaxis.set_ticks(np.arange(0, featuredata[features[7]].max() + 3, 1))
			ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax8.set_xlim(17.6, 20.3);
			ax8.set_ylim(-0.5, len(data[class_])+scale)
			ax8.set_title(labels[7])

			print(featuredata[features[8]])
			ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='European', markersize=3)
			ax9.plot(comparisondata[features[8]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax9.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[8]]):
				if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[8]]):
				if(y == comparisondata[features[8]].max() or y == comparisondata[features[8]].min()):
					ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax9.xaxis.set_ticks(np.arange(0, featuredata[features[8]].max() + 4, 1))
			ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax9.set_xlim(20, 24.2);
			ax9.set_ylim(-0.5, len(data[class_])+scale)
			ax9.set_title(labels[8])
			
			print(featuredata[features[9]])
			ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='European', markersize=3)
			ax10.plot(comparisondata[features[9]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax10.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[9]]):
				if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[9]]):
				if(y == comparisondata[features[9]].max() or y == comparisondata[features[9]].min()):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax10.xaxis.set_ticks(np.arange(0, featuredata[features[9]].max() + 3, 1))
			ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
			ax10.set_xlim(12, 14);
			ax10.set_ylim(-0.5, len(data[class_])+scale)
			ax10.set_title(labels[9])

			print(featuredata[features[10]])
			ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='European', markersize=3)
			ax11.plot(comparisondata[features[10]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax11.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[10]]):
				if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[10]]):
				if(y == comparisondata[features[10]].max() or y == comparisondata[features[10]].min()):
					ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax11.xaxis.set_ticks(np.arange(0, featuredata[features[10]].max() + 6, 0.5))
			ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax11.set_xlim(4, 5.8);
			ax11.set_ylim(-0.5, len(data[class_])+scale)
			ax11.set_title(labels[10])
			

			print(featuredata[features[11]])
			ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='European', markersize=3)
			ax12.plot(comparisondata[features[11]], comparisondata[class_], 'rx', label='non-European', markersize=3)
			ax12.grid(alpha=0.5, linestyle=':')
			for x, y in enumerate(featuredata[features[11]]):
				if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
					if(x == 1):
						ax12.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
					else:
						ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			for x, y in enumerate(comparisondata[features[11]]):
				if(y == comparisondata[features[11]].max() or y == comparisondata[features[11]].min()):
					if(x == 1):
						ax12.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
					else:
						ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
			ax12.xaxis.set_ticks(np.arange(0, featuredata[features[11]].max() + 1, 0.2))
			ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
			ax12.set_xlim(3.1, 3.9);
			ax12.set_ylim(-0.5, len(data[class_])+scale)
			ax12.set_title(labels[11])

			for ax in fig.get_axes():
				ax.tick_params(axis='x', labelsize=7)
				ax.tick_params(axis='y', labelsize=7)
	else:
		scale = -0.4

		fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, sharey=True, figsize=(8,4))

		print(featuredata[features[0]].min(), featuredata[features[0]].max(), comparisondata[features[0]].min(), comparisondata[features[0]].max())
		ax1.plot(featuredata[features[0]], featuredata[class_], '.', label='European', markersize=3)
		ax1.plot(comparisondata[features[0]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		#ax1.hlines(y=data[class_], xmin=0, xmax=featuredata[features[0]], color='firebrick', alpha=0.7, linewidth=10)
		ax1.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[0]]):
			if(y == featuredata[features[0]].max() or y == featuredata[features[0]].min()):
				if(x == 0):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
				else:
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[0]]):
			if(y == comparisondata[features[0]].max() or y == comparisondata[features[0]].min()):
				if(x == 0):
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
				else:
					ax1.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax1.xaxis.set_ticks(np.arange(0, 2.8, 0.1))
		ax1.set_xlim(1.65, 2.2);
		ax1.set_ylim(-0.5, len(data[class_])+scale)
		ax1.set_title(labels[0])
		#ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), prop={'size': 6})
		ax1.legend(loc='lower center', bbox_to_anchor=(-0.6,0.9), prop={'size': 6})


		

		print(featuredata[features[1]].min(), featuredata[features[1]].max(), comparisondata[features[1]].min(), comparisondata[features[1]].max())
		ax2.plot(featuredata[features[1]], featuredata[class_], '.', label='European', markersize=3)
		ax2.plot(comparisondata[features[1]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax2.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[1]]):
			if(y == featuredata[features[1]].max() or y == featuredata[features[1]].min()):
				ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[1]]):
			if(y == comparisondata[features[1]].max() or y == comparisondata[features[1]].min()):
				ax2.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax2.xaxis.set_ticks(np.arange(0, 0.6, 0.02))
		ax2.set_xlim(0.4, 0.485);
		ax2.set_ylim(-0.5, len(data[class_])+scale)
		ax2.set_title(labels[1])

		print(featuredata[features[2]].min(), featuredata[features[2]].max(), comparisondata[features[2]].min(), comparisondata[features[2]].max())
		ax3.plot(featuredata[features[2]], featuredata[class_], '.', label='European', markersize=3)
		ax3.plot(comparisondata[features[2]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax3.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[2]]):
			if(y == featuredata[features[2]].max() or y == featuredata[features[2]].min()):
				if(x == 0):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
				else:
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[2]]):
			if(y == comparisondata[features[2]].max() or y == comparisondata[features[2]].min()):
				if(x == 0):
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
				else:
					ax3.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax3.xaxis.set_ticks(np.arange(0, 70, 6))
		ax3.set_xlim(10, 30);
		ax3.set_ylim(-0.5, len(data[class_])+scale)
		ax3.set_title(labels[2])

		print(featuredata[features[3]].min(), featuredata[features[3]].max(), comparisondata[features[3]].min(), comparisondata[features[3]].max())
		ax4.plot(featuredata[features[3]], featuredata[class_], '.', label='European', markersize=3)
		ax4.plot(comparisondata[features[3]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax4.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[3]]):
			if(y == featuredata[features[3]].max() or y == featuredata[features[3]].min()):
				ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[3]]):
			if(y == comparisondata[features[3]].max() or y == comparisondata[features[3]].min()):
				ax4.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax4.xaxis.set_ticks(np.arange(0, 5, 0.1))
		ax4.set_xlim(4.35, 4.75);
		ax4.set_ylim(-0.5, len(data[class_])+scale)
		ax4.set_title(labels[3])

		print(featuredata[features[4]].min(), featuredata[features[4]].max(), comparisondata[features[4]].min(), comparisondata[features[4]].max())
		ax5.plot(featuredata[features[4]], featuredata[class_], '.', label='European', markersize=3)
		ax5.plot(comparisondata[features[4]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax5.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[4]]):
			if(y == featuredata[features[4]].max() or y == featuredata[features[4]].min()):
				if(x == 0):
					ax5.text(y-0.2, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
				else:
					ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[4]]):
			if(y == comparisondata[features[4]].max() or y == comparisondata[features[4]].min()):
				if(x == 0):
					ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
				else:
					ax5.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax5.xaxis.set_ticks(np.arange(0, 30, 2))
		ax5.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		ax5.set_xlim(22.5, 29.5);
		ax5.set_ylim(-0.5, len(data[class_])+scale)
		ax5.set_title(labels[4])

		print(featuredata[features[5]].min(), featuredata[features[5]].max(), comparisondata[features[5]].min(), comparisondata[features[5]].max())
		ax6.plot(featuredata[features[5]], featuredata[class_], '.', label='European', markersize=3)
		ax6.plot(comparisondata[features[5]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax6.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[5]]):
			if(y == featuredata[features[5]].max() or y == featuredata[features[5]].min()):
				ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[5]]):
			if(y == comparisondata[features[5]].max() or y == comparisondata[features[5]].min()):
				ax6.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax6.xaxis.set_ticks(np.arange(0, 15, 2))
		ax6.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		ax6.set_xlim(6, 10.5);
		ax6.set_ylim(-0.5, len(data[class_])+scale)
		ax6.set_title(labels[5])

		print(featuredata[features[6]].min(), featuredata[features[6]].max(), comparisondata[features[6]].min(), comparisondata[features[6]].max())
		ax7.plot(featuredata[features[6]], featuredata[class_], '.', label='European', markersize=3)
		ax7.plot(comparisondata[features[6]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax7.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[6]]):
			if(y == featuredata[features[6]].max() or y == featuredata[features[6]].min()):
				ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[6]]):
			if(y == comparisondata[features[6]].max() or y == comparisondata[features[6]].min()):
				ax7.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax7.xaxis.set_ticks(np.arange(0, 10, 0.5))
		ax7.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
		ax7.set_xlim(2.9, 4.1);
		ax7.set_ylim(-0.5, len(data[class_])+scale)
		ax7.set_title(labels[6])

		print(featuredata[features[7]].min(), featuredata[features[7]].max(), comparisondata[features[7]].min(), comparisondata[features[7]].max())
		ax8.plot(featuredata[features[7]], featuredata[class_], '.', label='European', markersize=3)
		ax8.plot(comparisondata[features[7]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax8.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[7]]):
			if(y == featuredata[features[7]].max() or y == featuredata[features[7]].min()):
				if(x == 0):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
				else:
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[7]]):
			if(y == comparisondata[features[7]].max() or y == comparisondata[features[7]].min()):
				if(x == 0):
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
				else:
					ax8.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax8.xaxis.set_ticks(np.arange(0, 55, 6))
		ax8.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		ax8.set_xlim(13, 44);
		ax8.set_ylim(-0.5, len(data[class_])+scale)
		ax8.set_title(labels[7])

		print(featuredata[features[8]].min(), featuredata[features[8]].max(), comparisondata[features[8]].min(), comparisondata[features[8]].max())
		ax9.plot(featuredata[features[8]], featuredata[class_], '.', label='European', markersize=3)
		ax9.plot(comparisondata[features[8]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax9.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[8]]):
			if(y == featuredata[features[8]].max() or y == featuredata[features[8]].min()):
				ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[8]]):
			if(y == comparisondata[features[8]].max() or y == comparisondata[features[8]].min()):
				ax9.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax9.xaxis.set_ticks(np.arange(0, 30, 3))
		ax9.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		ax9.set_xlim(12, 24.5);
		ax9.set_ylim(-0.5, len(data[class_])+scale)
		ax9.set_title(labels[8])
		
		print(featuredata[features[9]].min(), featuredata[features[9]].max(), comparisondata[features[9]].min(), comparisondata[features[9]].max())
		ax10.plot(featuredata[features[9]], featuredata[class_], '.', label='European', markersize=3)
		ax10.plot(comparisondata[features[9]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax10.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[9]]):
			if(y == featuredata[features[9]].max() or y == featuredata[features[9]].min()):
				if(x == 0):
					ax10.text(y+0.3, x+0.3, round(y, 2), horizontalalignment='right', fontsize=5)
				else:
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[9]]):
			if(y == comparisondata[features[9]].max() or y == comparisondata[features[9]].min()):
				if(x == 0):
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='left', fontsize=5)
				else:
					ax10.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax10.xaxis.set_ticks(np.arange(0, 40, 4))
		ax10.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		ax10.set_xlim(11, 22);
		ax10.set_ylim(-0.5, len(data[class_])+scale)
		ax10.set_title(labels[9])

		print(featuredata[features[10]].min(), featuredata[features[10]].max(), comparisondata[features[10]].min(), comparisondata[features[10]].max())
		ax11.plot(featuredata[features[10]], featuredata[class_], '.', label='European', markersize=3)
		ax11.plot(comparisondata[features[10]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax11.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[10]]):
			if(y == featuredata[features[10]].max() or y == featuredata[features[10]].min()):
				ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[10]]):
			if(y == comparisondata[features[10]].max() or y == comparisondata[features[10]].min()):
				ax11.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax11.xaxis.set_ticks(np.arange(0, 10, 0.5))
		ax11.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
		ax11.set_xlim(4.2, 5.7);
		ax11.set_ylim(-0.5, len(data[class_])+scale)
		ax11.set_title(labels[10])
		

		print(featuredata[features[11]].min(), featuredata[features[11]].max(), comparisondata[features[11]].min(), comparisondata[features[11]].max())
		ax12.plot(featuredata[features[11]], featuredata[class_], '.', label='European', markersize=3)
		ax12.plot(comparisondata[features[11]], comparisondata[class_], 'rx', label='non-European', markersize=3)
		ax12.grid(alpha=0.5, linestyle=':')
		for x, y in enumerate(featuredata[features[11]]):
			if(y == featuredata[features[11]].max() or y == featuredata[features[11]].min()):
				ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		for x, y in enumerate(comparisondata[features[11]]):
			if(y == comparisondata[features[11]].max() or y == comparisondata[features[11]].min()):
				ax12.text(y, x+0.3, round(y, 2), horizontalalignment='center', fontsize=5)
		ax12.xaxis.set_ticks(np.arange(0, 7, 0.3))
		ax12.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
		ax12.set_xlim(3.25, 4.21);
		ax12.set_ylim(-0.5, len(data[class_])+scale)
		ax12.set_title(labels[11])

		for ax in fig.get_axes():
			ax.tick_params(axis='x', labelsize=7)
			ax.tick_params(axis='y', labelsize=7)


	plt.tight_layout()
	plt.savefig('visualization/'+source+'_plot_'+class_, dpi=300)
	plt.close()

def plotConfusionMatrix(source):
	origin = [
			'Native',
			'NonNative'
		]

	if(source == 'Combined' or source == 'CombinedNE'):
		classes = ['Origin', 'Language', 'Language_Family']
		lang = [
				'Bulgarian',
				'Croatian',
				'Czech',
				'Dutch',
				'English',
				'Finnish',
				'French', 
				'German', 
				'Greek', 
				'Indian',
				'Italian',
				'Japanese',
				'Lithuanian',
				'Norwegian',
				'Polish',
				'Portuguese',
				'Romanian',
				'Russian', 
				'Serbian',
				'Slovene',
				'Spanish',
				'Swedish',
				'Turkish'
				]
		family = [
				'Balto-Slavic',
				'Germanic',
				'Greek',
				'Indo-Aryan',
				'Japonic',
				'Native',
				'Romance',
				'Turkic',
				
		]
	elif(source == 'Twitter'):
		classes = ['Origin', 'Language_Family', 'Language']
		lang = [
				'English',
				'French', 
				'German', 
				'Greek', 
				'Indian',
				'Japanese',
				'Russian', 
				'Turkish'
				]
		family = [
			'Balto-Slavic',
			'Germanic',
			'Greek',
			'Indo-Aryan',
			'Japonic',
			'Native',
			'Romance',
			'Turkic'
		]
	else:
		classes = ['Origin', 'Language', 'Language_Family']
		lang = [
				'Bulgarian',
				'Croatian',
				'Czech',
				'Dutch',
				'English',
				'Finnish',
				'French', 
				'German', 
				'Italian',
				'Lithuanian',
				'Norwegian',
				'Polish',
				'Portuguese',
				'Romanian',
				'Russian', 
				'Serbian',
				'Slovene',
				'Spanish',
				'Swedish'
				]
		family = [
			'Balto-Slavic',
			'Germanic',
			'Native',
			'Romance'
		]
	featuresets = ['Normal', 'TFIDF']
	models = ['RandomForest', 'Pipeline', 'LogisticRegression', 'SVM']
	fileList = []

	for x in classes:
		for feature in featuresets:
			for model in models:
				filename = 'classification/classification_report_'+source+'_'+x+'_'+feature+'_'+model+'.csv'
				fileList.append([filename,x,feature,model])

	output = pd.DataFrame({
			'RandomForest':{
				'Normal':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				},
				'TFIDF':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				}
			},
			'Pipeline':{
				'Normal':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				},
				'TFIDF':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				}
			},
			'LogisticRegression':{
				'Normal':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				},
				'TFIDF':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				}
			},
			'SVM':{
				'Normal':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				},
				'TFIDF':{
					'Language':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0 ,'std':0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Language_Family':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					},
					'Origin':{
							'accuracy': {'mean': 0,'median': 0,'std': 0},
							'f1_macro': {'mean': 0,'median': 0,'std': 0},
							'precision': {'mean':0,'median': 0,'std': 0},
							'cv': {'mean':0,'median': 0 ,'std':0}
					}
				}
			}
			})
	for file in fileList:
		print(file[1])
		print(file)
		if(file[1] == 'Origin'):
			labels = origin
			title = 'Result for Origin prediction'
		elif(file[1] == 'Language'):
			title = 'Result for Language prediction'
			labels = lang
		elif(file[1] == 'Language_Family'):
			labels = family
			title = 'Result for Language Family prediction'
		data = pd.read_csv(file[0], header=None, sep=',', skiprows=1)
		data.columns = ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'prediction', 'actual', 'cv']
		data = data[data.accuracy.str.contains('accuracy') == False]

		scores = ['accuracy', 'f1_macro', 'precision', 'cv']
		values = ['prediction', 'actual']


#
		#accuracy = pd.to_numeric(data['accuracy'],downcast='float')
		#f1 = pd.to_numeric(data['f1_macro'],downcast='float')
		#prec = pd.to_numeric(data['precision'],downcast='float')

		for score in scores:
			output[file[3]][file[2]][file[1]][score]['mean'] = pd.to_numeric(data[score],downcast='float').mean()
			output[file[3]][file[2]][file[1]][score]['median'] = pd.to_numeric(data[score],downcast='float').median()
			output[file[3]][file[2]][file[1]][score]['std'] = pd.to_numeric(data[score],downcast='float').std()
		
		
		





		y_true = []
		for x in data['actual'].values:

			x = x.split()
			#print(x[0],x[-1])
			try:
				#print(x[0])
				x[0] = x[0][1]
			except:
				#print(x[0])
				#print(x)
				x.pop(0)
				#print(x)
			x[-1] = x[-1][0]
			x = [int(s) for s in x]
			y_true.append(x)
		y_true = list(chain.from_iterable(y_true))
		#print(y_true)
		y_pred = []
		for x in data['prediction'].values:
			x = x.split()
			#print(x)
			try:
				x[0] = x[0][1]
			except:
				x.pop(0)
			x[-1] = x[-1][0]
			x = [int(s) for s in x]
			y_pred.append(x)
		y_pred = list(chain.from_iterable(y_pred))

		if(source == 'Twitter'):
			fig, ax = plt.subplots(figsize=(5, 5))
		elif(source == 'Combined'):
			if(file[1] == 'Language'):
				fig, ax = plt.subplots(figsize=(12, 12))
			else:
				fig, ax = plt.subplots(figsize=(5, 5))
		else:
			if(file[1] == 'Language'):
				fig, ax = plt.subplots(figsize=(9, 9))
			else:
				fig, ax = plt.subplots(figsize=(5, 5))


		array = confusion_matrix(y_true,y_pred)
		#print(array)
		df_cm = pd.DataFrame(array, index = labels,
				  columns = labels)
		sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt="d", ax=ax, lw=0.5)
		#sn.set(font_scale=1.4) # for label size
		ax.set(xlabel='Predicted label', ylabel='Actual label')
		#ax.set_title(title)
		plt.tight_layout()
		#plt.show()
		plt.savefig('visualization/'+source+'_'+file[1]+'_'+file[2]+'_'+file[3], dpi=300)
		plt.close()

	#output.to_csv('visualization/'+source+'_'+file[1]+'_'+file[2]+'_'+file[3]+'.csv')
	fields = ['Model', 'Features', 'Class', 'Score', 'Mean', 'Median', 'STD']
	scores = ['accuracy', 'f1_macro', 'precision', 'cv']
	with open("visualization/scores_"+source+".csv", "w") as f:
		w = csv.DictWriter(f, fields)
		w.writeheader()
		for model in models:
			#print(model)
			for feature in featuresets: 
				#print(feature)
				for clss in classes:
					#print(clss)
					for score in scores:
						#print(score)
						output[model][feature][clss][score]
						w.writerow({'Model':model, 'Features':feature, 'Class':clss, 'Score':score, 'Mean':output[model][feature][clss][score]['mean'], 'Median':output[model][feature][clss][score]['median'], 'STD':output[model][feature][clss][score]['std']})





if __name__ == "__main__":
	plotConfusionMatrix(sys.argv[1])
	#print("Creating plot "+sys.argv[1])
	#visualizeData(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])