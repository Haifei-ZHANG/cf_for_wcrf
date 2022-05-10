# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:44:57 2022

@author: Haifei Zhang
"""


import numpy as np
import copy


class RFCFExplainer:
    def __init__(self, wcrf, train_set, feature_names, bab=True, protected_features=None):
        self.bab = bab
        self.model = wcrf
        self.rf = wcrf.model
        self.s = wcrf.s
        self.classes = wcrf.model.classes_
        self.n_classes = len(self.classes)
        self.train_set = train_set
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.protected_features = protected_features
        self.extract_rules()
        
    def extract_rules(self):
        '''
        extract rules of trees, stocked as np.array, including lower and upper bound of each feature,
        probabilities array, and prediction of rule
        meanwhile, it gets all split points for each feature
        '''
        rules = np.zeros(2 * self.n_features + 2*len(self.classes) + 1)
        feature_depth = np.zeros(self.n_features)
        
        for tree in self.rf.estimators_:
            
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            value = tree.tree_.value.reshape((-1, self.n_classes))
            
            node_depth = np.zeros(len(feature))
            rules_in_tree = np.zeros(2 * self.n_features + 2*self.n_classes + 1)
            rules_stack = [[[-0.000001, 1] for i in range(self.n_features)]]

            for i in range(n_nodes):
                is_internal_node = (children_left[i] != children_right[i])
                if is_internal_node:
                    parent_rule = rules_stack.pop()
                    left_rule = copy.deepcopy(parent_rule)
                    right_rule = copy.deepcopy(parent_rule)
                    rules_stack.append(right_rule)
                    rules_stack.append(left_rule)
                    node_depth[children_left[i]] = node_depth[i] + 1
                    node_depth[children_right[i]] = node_depth[i] + 1
                    rules_stack[-1][feature[i]][1] = threshold[i]
                    rules_stack[-2][feature[i]][0] = threshold[i]
                    

                else:
                    # probabilities = (value[i]/sum(value[i])).round(5)
                    N_S = sum(value[i]) + self.s
                    n_samples = value[i].repeat(2)
                    n_samples[1:2*len(self.classes):2] = n_samples[1:2*self.n_classes:2] + self.s
                    probabilities = n_samples/N_S
                    if self.n_classes == 2:
                        if probabilities[0] >= 0.5:
                            prediction = 0
                        elif probabilities[1] <= 0.5:
                            prediction = 1
                        else:
                            prediction = -1
                    rule_to_add = np.array(rules_stack.pop()).flatten()
                    for k in range(len(probabilities)):
                        rule_to_add = np.append(rule_to_add, probabilities[k])
                    rule_to_add = np.append(rule_to_add, prediction)
                    rules_in_tree = np.vstack((rules_in_tree, rule_to_add))
            
            for f in range(self.n_features):
                feature_depth[f] += node_depth[feature==f].sum()
            
            rules = np.vstack((rules, rules_in_tree[1:,:]))
        
        self.feature_depth = feature_depth
        self.dimension_check_order = feature_depth.argsort()
        
        self.rules = rules[1:,:]
        
        self.split_points = {}
        for d in range(self.n_features):
            splits_points_d = np.unique(self.rules[:,2*d:2*d+2].reshape((1,-1)))
            splits_points_d.sort()
            self.split_points[d] = splits_points_d

            
    def filter_infeasible_rules(self, factual, protected_features, mo_dist):
        '''
        remove infeasible rules according to the list of protected features
        feasible rules must contain the values of protected features
        '''
        to_delete_columns = []
        feasible_rules = self.rules.copy()

        factual_rules_dist, _ = self.calculate_point2rules_dists(factual, feasible_rules)
        choice = factual_rules_dist[0] <= mo_dist
        feasible_rules = feasible_rules[choice]

        
        if protected_features is None:
            return feasible_rules
        else:
            for feature in protected_features:
                d = self.feature_names.index(feature)
                to_delete_columns.append(2*d)
                to_delete_columns.append(2*d+1)
                choise = (feasible_rules[:, 2*d] < factual[d]) * (factual[d] <= feasible_rules[:,2*d+1])
                feasible_rules = feasible_rules[choise,:]

            feasible_rules = np.delete(feasible_rules, to_delete_columns, axis=1)
        
        return feasible_rules
    
        
    def build_split_points(self, feasible_rules, feasible_D):
        '''
        obtain the split points for each dimension of feasible rules
        '''
        feasible_split_points = {}
        for d in range(feasible_D):
            splits_points_d = np.unique(feasible_rules[:,2*d:2*d+2].reshape((1,-1)))
            splits_points_d.sort()
            feasible_split_points[d] = splits_points_d
        
        return feasible_split_points
        
    
    
    def build_intervals(self, feasible_rules, feasible_split_points, feasible_D):
        '''
        build intervals for each feature according to the feasible split pokits
        the index of rules in which the interval is contained are also obtained
        '''
        rules_intersect_intervals = {}
        for d in range(feasible_D):
            rules_intersect_intervals[d] = {}
            rules_d = feasible_rules[:, 2*d:2*d+2]
            for i in range(len(feasible_split_points[d])-1):
                interval = (feasible_split_points[d][i], feasible_split_points[d][i+1])
                is_rules_intersect_interval = np.where(np.maximum(interval[0], rules_d[:,0]) < np.minimum(interval[1], rules_d[:,1]))[0]
                rules_intersect_intervals[d][interval] = is_rules_intersect_interval
        self.rules_intersect_intervals = rules_intersect_intervals
        return rules_intersect_intervals
    
    
    def reconstract_cf(self, factual_original, cf, actionable_features_index):
        '''
        reconstract to complete cf, the values of protected features equal to original values
        '''
        for i in range(len(actionable_features_index)):
            if actionable_features_index[i]:
                factual_original[i] = cf[0]
                cf = np.delete(cf, 0)
        
        return factual_original
                

    
    def extract_cf(self, factual, objective_class=None, protected_features=None):
        '''
        extract closest counterfactual
        here, factual should be a matrix of n*D
        '''
        if factual.ndim > 1:
            factual = factual[0]
        self.objective_class = objective_class
        factual_original = copy.deepcopy(factual)
        if protected_features is None:
            protected_features = self.protected_features
        
        mo, mo_dist = self.find_mo(factual, objective_class, protected_features)
        
        feasible_rules= self.filter_infeasible_rules(factual, protected_features, mo_dist)
        
        if protected_features is not None:
            actionable_features_index = np.ones(self.n_features, dtype='bool')
            for i in range(self.n_features):
                if self.feature_names[i] in protected_features:
                    actionable_features_index[i] = False

            feasible_D = sum(actionable_features_index)
            factual = factual[actionable_features_index]
        else:
            feasible_D = self.n_features
        
        cf = self.reconstract_cf(factual_original, mo, actionable_features_index)
        
        
        feasible_split_points = self.build_split_points(feasible_rules, feasible_D)
        
#         self.feasible_rules = self.rules
#         self.feasible_D = self.n_features
#         self.feasible_split_points = self.split_points
        
        rules_intersect_intervals = self.build_intervals(feasible_rules, feasible_split_points, feasible_D)
        
        
        intervals_check_order = {}
        intervals = {}
        for d in range(feasible_D):
            intervals[d] = list(rules_intersect_intervals[d].keys())
            intervals_cur = np.array(intervals[d])
            intervals_dists = np.maximum(np.maximum(0, intervals_cur[:,0]-factual[d]), np.maximum(0, factual[d]-intervals_cur[:,1]))
            intervals_check_order[d] = np.argsort(intervals_dists)
            interval = intervals[d][intervals_check_order[d][0]]
        
        cur_d = 0
        dist_inf = feasible_D
        dimension_dist = np.zeros(feasible_D)
        parent_region = np.ones(feasible_D*2)
        index_rules_in_regions = {}
        n_checked_intervals = np.zeros(feasible_D, dtype='int32')
        while True:
            if cur_d < 0:
                if protected_features is not None:
                    cf = self.reconstract_cf(factual_original, cf, actionable_features_index)
                return cf, dist_inf
            elif n_checked_intervals[cur_d] == len(intervals_check_order[cur_d]):
                n_checked_intervals[cur_d] = 0
                cur_d  -= 1
            else:
#                 if self.bab:
                to_check_interval_index = intervals_check_order[cur_d][n_checked_intervals[cur_d]]
                interval = intervals[cur_d][to_check_interval_index]

                if interval[0] < factual[cur_d] <= interval[1]:
                    dimension_dist[cur_d] = 0
                else:
                    dimension_dist[cur_d] = min(abs(interval[0] - factual[cur_d]), abs(interval[1] - factual[cur_d]))

                if sum(dimension_dist[:cur_d+1]) > dist_inf:
                    n_checked_intervals[cur_d] = 0
                    cur_d -= 1
                else:
                    if cur_d==0:
                        index_rules_in_regions[cur_d] = rules_intersect_intervals[cur_d][interval]
                    else:
                        index_rules_in_regions[cur_d] = np.intersect1d(index_rules_in_regions[cur_d-1], rules_intersect_intervals[cur_d][interval])
                        
                    if cur_d < feasible_D - 1:
                        parent_region[2*cur_d] = interval[0]
                        parent_region[2*cur_d+1] = interval[1]
                        n_checked_intervals[cur_d] += 1
                        cur_d += 1
                    else:
                        probabilities = feasible_rules[index_rules_in_regions[cur_d], 2*feasible_D:-1]
                        probabilities_aggr = np.zeros_like(probabilities, dtype='bool')
                        probabilities_aggr[:,:2*self.n_classes:2] = probabilities[:,:2*self.n_classes:2] >= 0.5
                        probabilities_aggr[:,1:2*self.n_classes:2] = probabilities[:,1:2*self.n_classes:2] >= 0.5
                        probabilities_aggr = np.mean(probabilities_aggr,axis=0)
                        if self.n_classes == 2:
                            if probabilities_aggr[0] >= 0.5:
                                prediction = 0
                            elif probabilities_aggr[1] <= 0.5:
                                prediction = 1
                            else:
                                prediction = -1
                        n_checked_intervals[cur_d] += 1
                        if prediction == objective_class:
                            parent_region[2*cur_d] = interval[0]
                            parent_region[2*cur_d+1] = interval[1]
                            dists, cf_candidates= self.calculate_point2rules_dists(factual, parent_region)
                            if dists.min() < dist_inf:
                                dist_inf = dists.min()
                                cf = cf_candidates[np.argmin(dists)]
                                print('Counterfactual:',cf)
                                print('distance=', dist_inf)


    def claculate_dist(self,m1, m2):
        ab = np.dot(m1, m2.T)
        a2 = np.sum(np.square(m1), axis=1).reshape((-1,1))
        b2 = np.sum(np.square(m2.T), axis=0).reshape((1,-1))
        to_sqrt = a2 + b2 - 2 * ab
        to_sqrt[to_sqrt<0] = 0
        return np.sqrt(to_sqrt)


    def calculate_points2points_dists(self):
        x = self.train_set
        ab = np.dot(x, x.T)
        a2 = np.sum(np.square(x), axis=1).reshape((-1,1))
        b2 = np.sum(np.square(x.T), axis=0).reshape((1,-1))
        to_sqrt = a2 + b2 - 2 * ab
        to_sqrt[to_sqrt<0] = 0
        self.point2point_dists = np.sqrt(to_sqrt)


    def calculate_point2rules_dists(self, point, rules):
        if rules.ndim == 1:
            rules = rules.reshape((1,-1))
        n_rules = len(rules)
        feasible_D = len(point)
        point2rule_dists = np.zeros(n_rules)
        
        points_in_rules = np.zeros((n_rules, len(point)))
        take_inf = point <= rules[:,0:2*feasible_D:2]
        take_sup = point > rules[:,1:2*feasible_D:2]
        take_self = np.array(1 - (take_inf + take_sup), dtype='bool')
        points_in_rules[take_inf] = rules[:,0:2*feasible_D:2][take_inf] + 0.0000001
        points_in_rules[take_sup] = rules[:,1:2*feasible_D:2][take_sup]
        point = point.reshape((1,-1))
        points_in_rules[take_self] = np.repeat(point, n_rules, axis=0)[take_self]
        ab = np.dot(point, points_in_rules.T)
        a2 = np.sum(np.square(point), axis=1).reshape((-1,1))
        b2 = np.sum(np.square(points_in_rules.T), axis=0).reshape((1,-1))
        to_sqrt = a2 + b2 - 2 * ab
        to_sqrt[to_sqrt<0] = 0
        point2rule_dists = np.sqrt(to_sqrt)
        
        return point2rule_dists, points_in_rules


    def find_mo(self, factual, objective_class=None, protected_features=None):
        if factual.ndim == 1:
            factual = factual.reshape((1,-1))
        x = self.train_set
        if protected_features is None:
            protected_features = self.protected_features
        if protected_features is not None:
            actionable_x_index = np.ones(len(x), dtype='bool')
            for feature in protected_features:
                feature_index = self.feature_names.index(feature)
                actionable_x_index = actionable_x_index * (x[:,feature_index]==factual[0][feature_index])
            x = x[actionable_x_index,:]
        y, _, _ = self.model.predict(x)
        if (objective_class==-1) or (objective_class is None):
            objective_class = 1
        
        x_of_objective_class = x[y==objective_class,:]
        
        if len(x_of_objective_class)==0:
            print('No such actionable counterfactual sample! Find no feature constrain MO!')
            return self.find_mo(factual, objective_class, protected_features=None)
        else:
            factual.reshape((1,-1))
            dists = self.claculate_dist(factual, x_of_objective_class)
            mo = x_of_objective_class[np.argmin(dists[0])]
            mo_dist = dists[0][np.argmin(dists[0])]
            return mo, mo_dist
    

    def find_one_change_cf(self, factual, objective_class=None, protected_features=None):
        if factual.ndim == 1:
            factual = factual.reshape((1,-1))
        if protected_features is None:
            protected_features = self.protected_features
        if protected_features is not None:
            feasible_features_index = []
            for feature in self.feature_names:
                if feature not in protected_features:
                    feasible_features_index.append(self.feature_names.index(feature))
        else:
            feasible_features_index = range(self.n_features)
        
        min_dist = 1
        cf = None
        d_to_change = -1
        for d in feasible_features_index:
            split_points = self.split_points[d]
            split_points[split_points<0] = 0
            n_split_points = len(split_points)
            insert_values = (split_points[1:n_split_points]+split_points[0:-1])/2
            ice_factuals = factual.repeat(n_split_points-1,axis=0)
            ice_factuals[:,d] = insert_values
            ice_predictions,ice_intervals,_ = self.model.predict(ice_factuals)
            ice_intervals = np.array(ice_intervals)
            
            if objective_class==0:
                condition = ice_intervals[:,1]<=0.5
            else:
                condition = ice_intervals[:,0]>=0.5
            if sum(condition)==0:
                continue
                
            ice_factuals = ice_factuals[condition]
            insert_values = insert_values[condition]
            ice_intervals = ice_intervals[condition]
            dists = abs(insert_values - factual[0,d])
            if min(dists) < min_dist:
                min_dist = min(dists)
                cf = ice_factuals[np.argmin(dists)]
                d_to_change = d
        if cf is None:
            cf, min_dist = self.find_mo(factual, objective_class,protected_features)
            d_to_change = factual[0]!=cf
            print('Use MO!')
            return cf, min_dist, d_to_change
        
        return cf, min_dist, d_to_change
    
    
    def find_initial_cf(self, factual, objective_class=None, protected_features=None):
        mo, mo_dist = self.find_mo(factual, objective_class,protected_features)
        occf, occf_dist =  self.find_one_change_cf(factual, objective_class,protected_features)
        if occf_dist < mo_dist:
            return occf, occf_dist
        else:
            return mo, mo_dist
        
            
    def build_tree(self):
            n_estimators = len(self.rf.estimators_)
            n_classes = len(self.classes)
            def creat_kfork_tree(parent_rules_index, currenr_d, rules=self.rules, intervals=self.rules_intersect_intervals, n_estimators =n_estimators, n_classes=n_classes):
                if len(parent_rules_index) == n_estimators:
                    probabilites = rules[parent_rules_index, -1-n_classes:-1].mean(axis=0)
                    prediction = np.argmax(probabilites)
                    return (probabilites, prediction)
                else:
                    sub_kfork_tree = {}
                    for interval_key in intervals[currenr_d].keys():
                        rules_of_interval_index = intervals[currenr_d][interval_key]
                        current_rules_index = np.intersect1d(parent_rules_index,rules_of_interval_index)
                        sub_kfork_tree[interval_key] = creat_kfork_tree(current_rules_index, currenr_d+1)
                    return sub_kfork_tree

            self.convered_tree = creat_kfork_tree(parent_rules_index=np.arange(len(self.rules)), currenr_d=0)
            