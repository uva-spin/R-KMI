def new_los(kin, pars, F_data, F_err, df):  #same inputs as before, df is the full dataset (in dataframe form)
    phii = [] #phii is list of lists, stores all phi data, has same length as kin
    for i in range(len(kin)): #this is for loop that populats phi with the data
        matching_rows = df[(df['k'] == kin[i,0]) & 
                           (df['QQ'] == kin[i,1]) & 
                           (df['xB'] == kin[i,2]) & 
                           (df['t'] == kin[i,3])]
        Phi = matching_rows.iloc[:, 6].to_numpy() 
        phii+=[Phi]
    
    F_dnn_list = [] #single list of all the F_dnn predicted data. size of F matches F_data
    for i in range(len(kin)):
        for j in range(len(phii[i])): 
            kinn = np.append(kin[i], phii[i][j]) #this is making new kin that has phi added onto it. steps through each
            #kin and then adds one each phi in the kin set. 
            #print(kinn)
            kinnn = tf.cast(kinn, pars.dtype)
            F_dnn_one = bkm10.total_xs(kinnn, pars) #calcualtes one of teh F_DNN using BKM. not sure if this will work right
            F_dnn_list +=[F_dnn_one] #adds the F to the lsit of F. 
    F_dnn_list = np.array(F_dnn_list)    #turns to numpy (not sure if needed)    
    F_dnn = tf.reshape(F_dnn_list, [-1]) # turns to tensor for gradients.   
    #kin = tf.cast(kin, pars.dtype)
    #F_dnn = tf.reshape(bkm10.total_xs(kin, pars), [-1])
    F_data = tf.cast(F_data, pars.dtype)
    F_err = tf.cast(F_err, pars.dtype)
    loss = tf.reduce_mean(tf.square( (F_dnn - F_data) / (F_err) ) )
    return loss  