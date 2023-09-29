import numpy as np

class GLVQ:
    def __init__(self, num_prototypes_per_class, initialization_type = 'mean',  learning_rate = 0.01):
 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha_zero = learning_rate


    
    def initialization(self, train_data, train_labels):
        if self.initialization_type == 'mean':
            """Prototype initialization: if number of prototypes is 1, prototype initialised is the mean
            if prototype is n>1, prototype initilised is the mean plus n-1 points closest to mean"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    list1.append(mu)#.astype(int))
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)

                    distances = [(mu-c)@(mu-c).T for c in class_data]
                    index = np.argsort(distances)
                    indices = index[1:self.num_prototypes]
                    prototype = class_data[indices]
                    r = np.vstack((mu, prototype))
                    list2.append(r)
                    ind = []
                    for j in range(self.num_prototypes ):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)#.flatten()   
                prototypes = M.reshape(num_protos,num_dims)

                P = np.array(prototypes)
            return np.array(new_labels).flatten(), P
        
        elif self.initialization_type == 'random':
            """Prototype initialization random: randomly chooses n points per class"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    random_int = np.random.choice(np.array(index))
                    prototype = train_data[random_int]
                    list1.append(prototype)
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                #regulate the prototypes, could also be done with GMM
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    random_integers = np.random.choice(np.array(index), size=self.num_prototypes)
                    prototype = train_data[random_integers]
                    list2.append(prototype)
                    ind = []
                    for j in range(self.num_prototypes):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)  
                prototypes = M.reshape(num_protos,num_dims)
                P = np.array(prototypes)
            return np.array(new_labels).flatten(), P 
        


    


    def sigmoid(self, x):
        import math
        denominator = 1 + math.exp(-x)
        return 1/denominator
    

    def abs_vec(self, x,y):
        r =  [(x[i] -y[i])**2 for i in range(len(x))]
        return np.array(r).sum()
    


    def update(self, data,prototypes, proto_labels,  labels, alpha):
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([self.abs_vec(xi, prototypes[j]) for j in range(len(prototypes)) if x_label == proto_labels[j]])
            d_a =dist_a.min()
            index_a = np.argmin(dist_a)
            dist_b = np.array([self.abs_vec(xi, prototypes[j]) for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_b = dist_b.min()
            index_b = np.argmin(dist_b)
            rel_dist = (d_a - d_b)/(d_a + d_b)
            f = self.sigmoid(rel_dist)
            prototypes[index_a] += alpha*(f*(1-f))*(np.divide(d_b, (d_a + d_b)**2))*(xi - prototypes[index_a])
            prototypes[index_b] -= alpha*(f*(1-f))*(np.divide(d_a, (d_a + d_b)**2))*(xi - prototypes[index_b])
            
            
            
        return self.prototypes
    

    def cost(self, data, labels, prototypes, proto_labels):
        l = []
    
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([self.abs_vec(xi, prototypes[j]) for j in range(len(prototypes)) if x_label == proto_labels[j]])
            d_a =dist_a.min()
            
            dist_b = np.array([self.abs_vec(xi, prototypes[j]) for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_b = dist_b.min()

            #if d_a - d_b < 0:
            rel_dist = (d_a - d_b)/(d_a + d_b)
            l.append(self.sigmoid(rel_dist))
        
        loss = np.array(l).sum()
        return (1/len(data))*loss
    def Pl_loss(self, unit, target_class):
        index = np.flatnonzero(self.proto_labels == target_class)[0]
        
        dist_a = np.array([self.abs_vec(unit, self.prototypes[index]) for j in range(len(self.prototypes)) if target_class == self.proto_labels[index]])
        d_a =dist_a.min()
        
        dist_b = np.array([self.abs_vec(unit, self.prototypes[j]) for j in range(len(self.prototypes)) if target_class != self.proto_labels[j]])
        d_b = dist_b.min()

        #if d_a - d_b < 0:
        rel_dist = (d_a - d_b)/(d_a + d_b)

        return self.sigmoid(rel_dist)


    def fit(self, data, labels, Epochs = 100, decay_scheme = True, plot_loss = False):
        import math
        import matplotlib.pyplot as plt
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        loss = []
        
        for epoch in range(Epochs):
            if decay_scheme == True:
                alpha = self.alpha_zero*(math.exp(-1*epoch/Epochs))
                self.prototypes = self.update(data,self.prototypes, self.proto_labels, labels, alpha)
                err = self.cost(data, labels, self.prototypes, self.proto_labels)
                
                loss.append(err)
            else:
                self.prototypes = self.update(data,self.prototypes, self.proto_labels, labels, self.alpha_zero)
                err = self.cost(data, labels, self.prototypes, self.proto_labels)
                
                loss.append(err)

            print(f'Epoch: {epoch}.......... Loss: {err}')
        if plot_loss == True:
            plt.plot(loss)
        return self.prototypes, self.proto_labels
        

    def predict_all(self, data, return_scores = False):

        """predict an array of instances""" 
        label = []
        #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)
        if return_scores == False:
            for i in range(data.shape[0]):
                xi = data[i]
                distances = np.array([np.linalg.norm(xi - p) for p in self.prototypes])
                index = np.argwhere(distances == distances.min())
                x_label = self.proto_labels[index]
                label.append(x_label)
            return np.array(label).flatten()
        else:
            predicted = []
            for i in range(len(data)):
                predicted.append(self.proba_predict(data[i]))
            return predicted 



    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = []
        for i in range(len(test_data)):
            predicted.append(self.predict(test_data[i]))
        val_acc = (np.array(predicted) == np.array(test_labels).flatten()).mean() * 100 
        return val_acc
    


    
    def predict(self, input):
        """predicts only one output at the time, numpy arrays only, 
        might want to convert"""
        
   


       
         
   
        distances = np.array([np.linalg.norm(input - p) for p in self.prototypes])
        index = np.argmin(distances)
        x_label = self.proto_labels[index]
        
        return x_label
    
    def proba_predict(self, input, softmax = False):
        """probabilistic prediction of a point by approximation of distances of a point to closest prototypes
        the argmin is the desired class"""
        scores = []
        closest_prototypes = []
        for i in np.unique(self.proto_labels):
            label_prototypes = self.prototypes[np.flatnonzero(self.proto_labels == i)]
            distances = np.array([np.linalg.norm(input - label_prototypes[j]) for j in range(label_prototypes.shape[0])])
            closest_prototype = label_prototypes[np.argmin(distances)]
            closest_prototypes.append(closest_prototype)
        dists = np.array([np.linalg.norm(input - prototype) for prototype in closest_prototypes])
        scores = np.array([d/dists.sum() for d in dists])
        if softmax == True:
            score = scores.copy()
            scores = [np.exp(-z)/(np.array(np.exp(-1*score)).sum()) for z in score]
        return scores 
    










class GRLVQ:
    def __init__(self, num_prototypes_per_class, initialization_type = 'mean',  prototype_update_learning_rate = 0.01, weight_update_learning_rate = 0.01):
 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha_zero = prototype_update_learning_rate
        self.eps_zero = weight_update_learning_rate


    def initialization(self, train_data, train_labels):
        if self.initialization_type == 'mean':
            """Prototype initialization: if number of prototypes is 1, prototype initialised is the mean
            if prototype is n>1, prototype initilised is the mean plus n-1 points closest to mean"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    list1.append(mu)#.astype(int))
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)

                    distances = [(mu-c)@(mu-c).T for c in class_data]
                    index = np.argsort(distances)
                    indices = index[1:self.num_prototypes]
                    prototype = class_data[indices]
                    r = np.vstack((mu, prototype))
                    list2.append(r)
                    ind = []
                    for j in range(self.num_prototypes ):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)#.flatten()   
                prototypes = M.reshape(num_protos,num_dims)

                P = np.array(prototypes)
            return np.array(new_labels).flatten(), P
        
        elif self.initialization_type == 'random':
            """Prototype initialization random: randomly chooses n points per class"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    random_int = np.random.choice(np.array(index))
                    prototype = train_data[random_int]
                    list1.append(prototype)
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                #regulate the prototypes, could also be done with GMM
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    random_integers = np.random.choice(np.array(index), size=self.num_prototypes)
                    prototype = train_data[random_integers]
                    list2.append(prototype)
                    ind = []
                    for j in range(self.num_prototypes):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)  
                prototypes = M.reshape(num_protos,num_dims)
                P = np.array(prototypes)
            return np.array(new_labels).flatten(), P 
        


    def weights(self, data):
        weight = np.full(data.shape[1], fill_value = 1/data.shape[1])
        
        return weight
    


    def sigmoid(self, x):
        import math
        denominator = 1 + np.exp(-x)
        return 1/denominator




    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x)) 




    def dist(self, x,y,w):
    
        r =  [(w[i]*(x[i] -y[i]))**2 for i in range(len(x))]
        f = np.array(r).sum()
        return f  
    


    def update(self, data,weight, proto_labels, prototypes ,alpha,eps, labels):
        import matplotlib.pyplot as plt
        
        
        
    
            
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([self.dist(xi, prototypes[j],weight) for j in range(len(prototypes)) if x_label == proto_labels[j]])
            d_a =dist_a.min()
            index_a = np.argmin(dist_a)
            dist_b = np.array([self.dist(xi, prototypes[j],weight) for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_b = dist_b.min()
            index_b = np.argmin(dist_b)
            rel_dist = (d_a - d_b)/(d_a + d_b)
            f = self.sigmoid(rel_dist)
            prototypes[index_a] += alpha*(f*(1-f))*(np.divide(d_b, (d_a + d_b)**2))*(xi - prototypes[index_a])
            prototypes[index_b] -= alpha*(f*(1-f))*(np.divide(d_a, (d_a + d_b)**2))*(xi - prototypes[index_b])
            weight  -= eps*self.sigmoid_prime((np.divide(d_b, (d_a + d_b)**2))*(xi - prototypes[index_a])**2 - (np.divide(d_a, (d_a + d_b)**2))*(xi - prototypes[index_b])**2)
            weight = weight.clip(min = 0)
            weight = weight/weight.sum()         
            
            
        return prototypes, weight
    


    def cost(self, data,prototypes, weight, labels, proto_labels):
        l = []
       
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([self.dist(xi,prototypes[j], weight) for j in range(len(prototypes)) if x_label == proto_labels[j]])
            d_a =dist_a.min()
            
            dist_b = np.array([self.dist(xi,prototypes[j], weight) for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_b = dist_b.min()
            
        
            
            rel_dist = (d_a - d_b)/(d_a + d_b)
            l.append(self.sigmoid(rel_dist).flatten())        
        loss = np.array(l).sum()
        return loss
    



        



    def fit(self, data, labels, Epochs = 100, decay_scheme = True, plot_loss = False):
        import math
        import matplotlib.pyplot as plt
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.weight = self.weights(data)
        loss = []
        
        for epoch in range(Epochs):
            if decay_scheme == True:
                alpha = self.alpha_zero*(math.exp(-1*epoch/Epochs))
                eps = self.eps_zero*(math.exp(-1*epoch/Epochs))
                self.prototypes, self.weight = self.update(data,self.weight, self.proto_labels, self.prototypes ,alpha, eps,labels)
                err = self.cost( data,self.prototypes, self.weight, labels, self.proto_labels)
                
                loss.append(err)
            else:
                self.prototypes, self.weight = self.update(data,self.weight, self.proto_labels, self.prototypes ,self.alpha_zero, self.eps_zero,labels)
                err = self.cost( data,self.prototypes, self.weight, labels, self.proto_labels)
                
                loss.append(err)

            print(f'Epoch: {epoch}.......... Loss: {err}')
        if plot_loss == True:
            plt.plot(loss)
        return self.prototypes, self.proto_labels, self.weight
    



    def predict_all(self, data, return_scores = False):

        """predict an array of instances""" 
        label = []
        #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)
        if return_scores == False:
            for i in range(data.shape[0]):
                xi = data[i]
                distances = np.array([np.linalg.norm(xi - p) for p in self.prototypes])
                index = np.argwhere(distances == distances.min())
                x_label = self.proto_labels[index]
                label.append(x_label)
            return np.array(label).flatten()
        else:
            predicted = []
            for i in range(len(data)):
                predicted.append(self.proba_predict(data[i]))
            return predicted 



    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = []
        for i in range(len(test_data)):
            predicted.append(self.predict(test_data[i]))
        val_acc = (np.array(predicted) == np.array(test_labels).flatten()).mean() * 100 
        return val_acc
    


    
    def predict(self, input):
        """predicts only one output at the time, numpy arrays only, 
        might want to convert"""
        
   


       
         
   
        distances = np.array([np.linalg.norm(input - p) for p in self.prototypes])
        index = np.argmin(distances)
        x_label = self.proto_labels[index]
        
        return x_label
    
    def proba_predict(self, input, softmax = False):
        """probabilistic prediction of a point by approximation of distances of a point to closest prototypes
        the argmin is the desired class"""
        scores = []
        closest_prototypes = []
        for i in np.unique(self.proto_labels):
            label_prototypes = self.prototypes[np.flatnonzero(self.proto_labels == i)]
            distances = np.array([np.linalg.norm(input - label_prototypes[j]) for j in range(label_prototypes.shape[0])])
            closest_prototype = label_prototypes[np.argmin(distances)]
            closest_prototypes.append(closest_prototype)
        dists = np.array([np.linalg.norm(input - prototype) for prototype in closest_prototypes])
        scores = np.array([d/dists.sum() for d in dists])
        if softmax == True:
            score = scores.copy()
            scores = [np.exp(-z)/(np.array(np.exp(-1*score)).sum()) for z in score]
        return scores 
    
        