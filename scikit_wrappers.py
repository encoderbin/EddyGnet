
import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection

from model import repre_utils
from model import triplet_loss
from model import causal_cnn
import time


class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder                
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):

        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):

        self.save_encoder(prefix_file)
        sklearn.externals.joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):

        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):

        self.load_encoder(prefix_file)
        self.classifier = sklearn.externals.joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_classifier(self, features, y):

        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        train_size = numpy.shape(features)[0]

        self.classifier = sklearn.svm.SVC(
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else numpy.inf,
            gamma='scale'
        )
        if train_size // nb_classes < 5 or train_size < 50 or self.penalty is not None:
            return self.classifier.fit(features, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                self.classifier, {
                    'C': [
                        0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                        numpy.inf
                    ],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [None]
                },
                cv=5, iid=False, n_jobs=5
            )
            if train_size <= 10000:
                grid_search.fit(features, y)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = sklearn.model_selection.train_test_split(
                    features, y,
                    train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
            return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):

        # Check if the given time series have unequal lengths
        print('X',X.shape)
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:                                          #有监督
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = repre_utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        train_length=len(train_generator)
        while i < self.nb_steps:
            i += 1
            if verbose:
                print('Epoch: ', epochs + 1)
                
            epoch_start_time = time.time()    
            iter = 0
            for batch in train_generator:            
                
#                 print(batch.shape)
#                 print('batch',flush=True)
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                loss.backward()
                self.optimizer.step()
                t2 = time.time()
                if iter%100==0:
                    print('train batch %s / %s, loss: %.2f' % (iter + 1, train_length, loss.item()))
                    print('time: %.2f'%(t2-epoch_start_time))   
                iter += 1
#             if i >= self.nb_steps:
#                 break
            epochs += 1
            # Early stopping strategy
#             if self.early_stopping is not None and y is not None and (
#                 ratio >= 5 and train_size >= 50
#             ):
#                 # Computes the best regularization parameters
#                 features = self.encode(X)
#                 self.classifier = self.fit_classifier(features, y)
#                 # Cross validation score
#                 score = numpy.mean(sklearn.model_selection.cross_val_score(
#                     self.classifier, features, y=y, cv=5, n_jobs=5
#                 ))
#                 count += 1
#                 # If the model is better than the previous one, update
#                 if score > max_score:
#                     count = 0
#                     found_best = True
#                     max_score = score
#                     best_encoder = type(self.encoder)(**self.params)
#                     best_encoder.double()
#                     if self.cuda:
#                         best_encoder.cuda(self.gpu)
#                     best_encoder.load_state_dict(self.encoder.state_dict())
#             if count == self.early_stopping:
#                 break

        # If a better model was found, use it
#         if found_best:
#             self.encoder = best_encoder

        return self.encoder

    def fit(self, X, y, save_memory=False, verbose=False):

        # Fitting encoder
        self.encoder = self.fit_encoder(
            X, y=y, save_memory=save_memory, verbose=verbose
        )

        # SVM classifier training
        features = self.encode(X)
        self.classifier = self.fit_classifier(features, y)

        return self

    def encode(self, X, batch_size=50):

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = repre_utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.encoder = self.encoder.train()
        print('encode_features:',features.shape)
        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):

        features = numpy.empty((
                numpy.shape(X)[0], self.out_channels,
                numpy.shape(X)[2] - window + 1
        ))
        masking = numpy.empty((
            min(window_batch_size, numpy.shape(X)[2] - window + 1),
            numpy.shape(X)[1], window
        ))
        compute_length = numpy.shape(X)[2]
        print('compute_length',compute_length)
        myi=0
        
        features_idt=X[:,:2,window-1:]
#         features_idt=torch.from_numpy(features_idt)
        print('features_idt',features_idt.shape)
        features_local=X[:,2:4,window-1:]
        print('features_local',features_local.shape)
        start_time = time.time()
        for b in range(numpy.shape(X)[0]):
            for i in range(math.ceil(
                (numpy.shape(X)[2] - window + 1) / window_batch_size)    # 分50组 1万  来处理
            ):
                train_start_time = time.time()
                for j in range(
                    i * window_batch_size+window-1,
                    min(
                        (i + 1) * window_batch_size+window-1,
                        numpy.shape(X)[2] )):
                    j0 = j - i * window_batch_size-window+1
                    masking[j0, :, :] = X[b, :, j-window+1: j +1]
                    
                    myi=myi+1
                    if myi % 10000 == 0:
                        print('compute %s / %s' % (myi, compute_length))
                    
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = numpy.swapaxes(
                    self.encode(masking[:j0 + 1,2:,:], batch_size=batch_size), 0, 1
                )
                print('encode_window_features:',features.shape)
                print('train time every whole data:%.2fs' % (time.time() - train_start_time), flush=True)
                print('total time:%.2fs' % (time.time() - start_time), flush=True)
                
        features= torch.from_numpy(features)       
        features_cat=torch.cat((features_idt, features), dim=1)        
        features_cat[:,2:4,:]=features_local
        return features_cat

    def predict(self, X, batch_size=50):

        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, batch_size=50):

        features = self.encode(X, batch_size=batch_size)
        return self.classifier.score(features, y)


class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):

    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = repre_utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


