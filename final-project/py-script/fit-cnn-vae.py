for label in train_data['label'].unique():
    print("Training PPCA for label: {}".format(label))
    x_train = train_data.loc[train_data['label']==label].copy()
    y_train = x_train.pop('label')
    train_input_fn = numpy_input_fn(
        x_train.values.astype(np.float32), 
        shuffle=True, 
        batch_size=1
    )
    estimator = train_cnn(train_input_fn)
    estimators.append((label, estimator))