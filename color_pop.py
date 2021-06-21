
'''
Creates the populations and the the learning model for the color spaces
'''

import random as rnd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import partition as part



class MLP(nn.Module):
    def __init__(self, number_of_points, max_categories):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.bn = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, max_categories)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return F.softmax(x, dim=1) 



class NetworkAgent():
    def __init__(self, max_categories, color_space):
        self.number_of_points = len(color_space.points)
        self.model = MLP(self.number_of_points, max_categories)
        self.color_space = color_space
        self.prev_generations = []
        self.prev_idxs = []

    def accuracy(self, test_x, test_y, batch_size=32):
        correct = 0
        total = 0
        with torch.no_grad():
            num_batches = int(len(test_x) / batch_size)

            for batch in range(num_batches):
                batch_x = test_x[batch*batch_size:(batch+1)*batch_size]
                batch_y = test_y[batch*batch_size:(batch+1)*batch_size]
                predictions = self.model(batch_x)
                
                for idx, i in enumerate(predictions):
                    if  torch.argmax(i) == batch_y[idx]:
                        correct += 1
                    total += 1

        return round(correct/total, 6)

    def predict(self, idxs):
        '''
        data = np.array([self.color_space.points[idx] for idx in idxs])

        with torch.no_grad():
            predictions = self.model(data)

        predictions_argmax = []
        for i in range(len(predictions)):
            predictions_argmax.append(np.argmax(predictions[i]))

        return data, np.array(predictions_argmax)
        '''
        data = np.array([self.color_space.points[idx] for idx in idxs])	
        predictions = np.array([self.color_space.labelled_pts[idx] for idx in idxs])	
    
        return data, predictions


    # Map the labels on the color space based on the trained model
    def learned_color_space(self, batch_size=32):
        test_x, test_y = self.make_train_bins(self.color_space, train=False)

        with torch.no_grad():
            num_batches = int(len(test_x) / batch_size)

            for batch in range(num_batches):
                batch_x = test_x[batch*batch_size:(batch+1)*batch_size]
                predictions = self.model(batch_x)

                for idx, i in enumerate(predictions):
                    prediction = int(torch.argmax(i))
                    idx_space = batch * batch_size + idx
                    self.color_space.relabel_point(idx_space, prediction)


    def shuffle(self, x, y):
        total = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(total)
        x = total[:, :len(total[0])-1]
        y = total[:, len(total[0])-1].astype(int)
        return x, y


    def from_bins_to_xy(self, points, bins):
        x = np.vstack([points[bins[label]] for label in bins])
        y = np.concatenate([[label]*len(bins[label]) for label in
                            bins]).astype(int)
        return x, y


    def make_train_bins(self, color_space, train=True):
        part = color_space.partition
        points = color_space.points

        train_split = 1
        train_bins = {label: part[label][:int(train_split*len(part[label]))] for
                      label in part}

        max_train_bin = max(len(train_bins[label]) for label in train_bins)

        if train:
            train_bins = {label: np.random.choice(train_bins[label], max_train_bin,
                                                  replace=True)
                          if 0 < len(train_bins[label]) < max_train_bin else
                          train_bins[label] for label in train_bins}

        train_x, train_y = self.from_bins_to_xy(points, train_bins)
        return train_x, train_y

    def learn(self, color_space, batch_size=32, num_epochs=3, shuffle_by_epoch=True):
        train_x, train_y = self.make_train_bins(color_space)

        optim = torch.optim.Adam(self.model.parameters())
        for epoch in range(num_epochs):
            if shuffle_by_epoch:
                train_x, train_y = self.shuffle(train_x, train_y)

            num_batches = int(len(train_x) / batch_size)
            for batch in range(num_batches):

                optim.zero_grad()
                # get net predictions
                batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                predictions = self.model(batch_x)
                batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
                # loss
                loss = F.cross_entropy(predictions, torch.tensor(batch_y)) # CHANGED was nll
                # back-propagate the loss
                loss.backward()
                optim.step()

        self.learned_color_space()



class Population:
    def __init__(self, size, max_categories, color_spaces):
        self.n_agents = len(color_spaces)
        self.max_categories = max_categories
        self.partitions = color_spaces

        self.agents = [NetworkAgent(max_categories, space) 
                        for space in color_spaces]

    

    def learn_from_population(self, parent_pop, bottleneck_size, 
                              num_epochs=6, shuffle_input=False):        

        parents = []
        for child in self.agents:
            parent_idx = rnd.randrange(len(parent_pop.agents))
            parents.append(parent_idx)
            parent = parent_pop.agents[parent_idx]

            child.prev_generations = copy.deepcopy(parent.prev_generations)
            child.prev_generations.append(copy.deepcopy(parent.color_space))

            child.prev_idxs = copy.deepcopy(parent.prev_idxs)
            child.prev_idxs.append(parent_idx)

            # pick the indexes of the color spaces that the learner will observe 
            # (with substitution)
            amt_points = int(parent.number_of_points * bottleneck_size)
            partition_idxs = np.random.randint(0, parent.number_of_points, size=(amt_points))
            partition_idxs = np.sort(partition_idxs)


            # make the parent produce the data for the sampled models
            parent_points, parent_predictions = parent.predict(partition_idxs)
            zeros = parent_points.shape[1]

            parent_space = part.Partition(parent_points, max(parent.color_space.labels)+1, 
                            np.zeros(zeros))
            for i in range(len(parent_space.points)):
                parent_space.relabel_point(i, parent_predictions[i])


            # shuffle each input model
            if shuffle_input:
                np.random.shuffle(partition_idxs)
            child.learn(parent_space, num_epochs=num_epochs)

        for agent in parent_pop.agents:
            agent.prev_generations = []

        return parents



if __name__ == '__main__':
    test_pop = Population(1, 7, './trial/')
    test_pop.learn_from_population()
    