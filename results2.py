aes_files = ['ake','aae','aam','aac']
wati_files = ['wke','wae','wam','wac']

for m in range(1,7):
    print "METRIC", m
    for file_name in aes_files:
    	print file_name
    	for i in range(2,11):
    		method_mean = []
    		for j in range(10):
    			with open('./out2/'+file_name+str(i)+'-'+str(j)) as f:
    	    			content = f.readlines()
    				method_mean.append(float(content[m]))
    		print sum(method_mean)/len(method_mean)

for m in range(1,7):
    print "METRIC", m
    for file_name in wati_files:
    	print file_name
    	for i in range(2,11):
    		method_mean = []
    		for j in range(10):
    			with open('./out2/'+file_name+str(i)+'-'+str(j)) as f:
    	    			content = f.readlines()
    				method_mean.append(float(content[m]))
    		print sum(method_mean)/len(method_mean)
