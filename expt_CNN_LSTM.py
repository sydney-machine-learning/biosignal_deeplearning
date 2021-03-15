# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, n_steps, n_length):
	# define model
	verbose, epochs, batch_size = 0, 25, 16
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = n_steps, n_length
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def summarize_results(scores):
    print(scores)
    best, m, s = max(scores), mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    print("Best: {}".format(best))
    
def run_experiment(trainX, trainy, testX, testy, repeats=10):
    scores = list()
    L = M = [(1,2500),(2,1250),(4,625),(5,500),(10,250),(20,125),(25,100),(50,50)]
    
    for data in L:
        print(data)
        for r in range(repeats):
            score = evaluate_model(trainX, trainy, testX, testy, data[0],data[1])
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        summarize_results(scores)