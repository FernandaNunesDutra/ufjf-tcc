aes_files = ['zout-ake','zout-aae','zout-aam','zout-aac']
wati_files = ['zout-wke','zout-wae','zout-wam','zout-wac']

for file_name in wati_files:
	print file_name
	for i in range(2,11):
		method_mean = []
		for j in range(10):
			with open(file_name+str(i)+'-'+str(j)) as f:
	    			content = f.readlines()
				method_mean.append(float(content[6]))
		print sum(method_mean)/len(method_mean)



		
