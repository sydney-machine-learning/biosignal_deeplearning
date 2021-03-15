def evaluate_model(trainX, trainy, testX, testy, nos):
	verbose, epochs, batch_size = 0, 15, 16
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(nos, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
 
# summarize scores
def summarize_results(scores):
    print(scores)
    best, m, s = max(scores), mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    print("Best: {}".format(best))
# run an experiment
def run_experiment(trainX, trainy, testX, testy, repeats=10):
    scores = list()
    for i in range(1,11):
        print("10*{}".format(i))
        for r in range(repeats):            
            score = evaluate_model(trainX, trainy, testX, testy,10*i)
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        summarize_results(scores)