aes_files = ['zout-ar']
wati_files = ['zout-wr']

for file_name in aes_files:
	print file_name
	for i in range(6):
		metric_mean = []
		for j in range(10):
			with open(file_name+'-'+str(j)) as f:
	    			content = f.readlines()
				metric_mean.append(float(content[i]))
		print sum(metric_mean)/len(metric_mean)


