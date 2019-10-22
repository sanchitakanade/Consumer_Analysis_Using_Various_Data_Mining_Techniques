#!/usr/bin/env python
# coding: utf-8

# ## Data mining Project

# ### Association rule mining(Apriori algorithm)

# In[18]:


# Data mining project
# Association rule mining(Apriori algorithm)
# In[4]:


# import packages for association mining

import pandas as pd   
import itertools


# In[17]:


# function to read data and preprocess data

def load_data():
    
    # read data from csv file
    read_customers_data=pd.read_csv("./BlackFriday.csv",index_col=False)
    
    # reduce data size by sampling
    # it takes less than a minute for this small amount of data
    customers_data=read_customers_data.sample(frac=0.1, random_state=1)
    
    # categorize attribute values for optimal rule generation
    customers_data['split_purchase'] = customers_data['Purchase'].apply(lambda x: "<5000 $" if x<=5000 else "5000-10000 $"                                    if x<=10000 else "10000-15000 $"                                    if x<=15000 else "> 15000 $")
    
    customers_data['occupation_category'] = customers_data['Occupation'].apply(lambda x: "Occupation 0-5" if x<=5                                            else "Occupation 6-10" if x<=10 else "Occupation 11-15"                                            if x<=15 else "Occupation 16-21")
    
    customers_data.replace({'Gender':{'F':'Female','M':'Male'}},inplace=True)
    
    customers_data.replace({'Marital_Status': {0: 'Single', 1: 'Married'}},inplace=True)
    
    customers_data['Age']=customers_data['Age'].astype(str)+" Age Group"
    
    customers_data=customers_data[['Age','split_purchase','Gender','Marital_Status','occupation_category']]
        
    customers_data.reset_index(drop=True, inplace=True)
    
    # calculate total number of records
    num_of_records = len(customers_data.index)
    
    return(customers_data,num_of_records)
   
    


# In[6]:



def get_feature_set(dataset):
    
    # function to convert rows into a list of strings called feature set
    feature_set=[]
    end=dataset.shape[1]+1
    
    # read each word in columns
    data_columns = [i for i in dataset.columns]
    
    # read each word in rows of dataset
    data_rows = [[i for i in row[1:end]] for row in dataset.itertuples()]
    
    feature_set = data_rows
    
    return feature_set
    


# In[7]:



def get_support_and_confidence():
    
    # function to generate input prompt for min support and min confidence
    min_sup = input('Enter minimum Support: ')
    min_conf = input('Enter minimum Confidence: ')

    # convert into integers
    min_sup = int(min_sup)
    min_conf = int(min_conf)
    
    return min_sup,min_conf
        

    


# In[8]:


def compute_first_set(feature_set):
    
    # function to generate first candidate set
    
    word_count_dict = {}
    candidate_set = []
    
    # count the number of occurences of each feature
    for line in feature_set:
        for word in line:
            if word not in word_count_dict:
                 word_count_dict[word] = 1
            else:
                 word_count_dict[word] = word_count_dict[word] + 1
    
    # convert feature and count into nested lists
    for key in word_count_dict:
        convert_list = []
        convert_list.append(key)
        candidate_set.append(convert_list)
        candidate_set.append(word_count_dict[key])

    return candidate_set


# In[9]:


def generate_itemsets(first_set, num_of_records, min_sup, feature_set, total_frequent_itemsets):
    
    # function to generate itemsets
    
    freq_templist = []
    removed_itemsets = []
    
    # loop through each item in candidate set
    for i in range(len(first_set)):
        
        # since list contains support count in even positions
        if i%2 != 0:
            # compute support percent of each itemset
            support = (first_set[i] * 1.0 / num_of_records) * 100
            
            # check if support of item is greater than min support
            if support >= min_sup:
                # add that item to a temporary list
                freq_templist.append(first_set[i-1])
                freq_templist.append(first_set[i])
            else:
                removed_itemsets.append(first_set[i-1])
                
    # add the above filtered lists to total frequent item set list
    for k in freq_templist:
        total_frequent_itemsets.append(k)
        #print(total_frequent_itemsets)

    if len(freq_templist) == 2 or len(freq_templist) == 0:
        freq_item_set = total_frequent_itemsets
       
        return freq_item_set

    else:
        
        generate_combinations(feature_set, removed_itemsets, freq_templist, num_of_records, min_sup,total_frequent_itemsets)


# In[10]:


def generate_combinations(feature_set,removed_itemsets, freq_templist, num_of_records, min_sup,total_frequent_itemsets):
    
    # function to generate pairs of combinations of the itemset
    
    feature_name = []
    combinations_set = []
    candidateSet = []
    
    for i in range(len(freq_templist)):
        if i%2 == 0:
            feature_name.append(freq_templist[i])
   
    
    for item in feature_name:
        
        temp_set = []
        
        item_index = feature_name.index(item)
        
        for i in range(item_index + 1, len(feature_name)):
            
            for j in item:
                if j not in temp_set:
                    temp_set.append(j)
                    
            for m in feature_name[i]:
                if m not in temp_set:
                    temp_set.append(m)
            # add combinations list to combination_set       
            combinations_set.append(temp_set)
            temp_set = []
            
    #print(combinations_set)
    sorted_set = []
    unique_set = []
    
    for i in combinations_set:
        sorted_set.append(sorted(i))
        
    for i in sorted_set:
        if i not in unique_set:
            unique_set.append(i)
            
    combinations_set = unique_set
    
    for item in combinations_set:
        count = 0
        for transaction in feature_set:
            if set(item).issubset(set(transaction)):
                count = count + 1
        if count != 0:
            candidateSet.append(item)
            candidateSet.append(count)
    # generate itemset based on min support from the combinations formed       
    generate_itemsets(candidateSet,num_of_records, min_sup,feature_set, total_frequent_itemsets)


# In[11]:


def generate_association_rules(total_set):
    
    # function to find combinations that can produce association rules
    
    association_rule = []
    
    for item in total_set:
        
        if isinstance(item, list):
            if len(item) != 0:
                length = len(item) - 1
                
                while length > 0:
                    
                    # generate combinations of final itemsets of different lengths
                    combinations = list(itertools.combinations(item, length))
                    temp = []
                    LHS = []
                    
                    for RHS in combinations:
                        # generate different combinations of same itemset
                        LHS = set(item) - set(RHS)
                        temp.append(list(LHS))
                        temp.append(list(RHS))
                        association_rule.append(temp)
                        temp = []
                        
                    length = length - 1
                    
    return association_rule


# In[12]:


def prune_rules_by_apriori(rules, feature_set, min_sup, min_conf,num_of_records):
    
    # function to apply apriori principle to filter rules
    ruleset_apriori = []
    
    for rule in rules:
        
        sup_count_A = 0
        support_percent_A = 0 #percentage support value
        sup_count_B = 0
        support_percent_B = 0
        sup_count_AandB = 0
        support_percent_AandB = 0
        
        # compute support count of left hand side of rule and right hand side of rule
        # and support of both items combined
        for transaction in feature_set:
            if set(rule[0]).issubset(set(transaction)):
                sup_count_A = sup_count_A + 1
            if set(rule[1]).issubset(set(transaction)):
                sup_count_B = sup_count_B + 1
            if set(rule[0] + rule[1]).issubset(set(transaction)):
                sup_count_AandB = sup_count_AandB + 1
                
        # calculate respective percentages of the above support values       
        support_percent_A = (sup_count_A * 1.0 / num_of_records) * 100
        support_percent_B = (sup_count_B * 1.0 / num_of_records) * 100
        support_percent_AandB = (sup_count_AandB * 1.0 / num_of_records) * 100
        confidence = (support_percent_AandB / support_percent_A) * 100
        
        # compute lift to check for positive correlation
        lift = confidence/support_percent_B
        
        # check confidence and lift values of rule
        # if lift>1 then the rule is positively correlated
        if ((confidence >= min_conf) & (lift>1)):
            
            support_of_A_list = "Support of A: " + str(round(support_percent_A, 2))
            support_of_B_list = "Support of B: " + str(round(support_percent_B, 2))
            support_of_AandB_list = "Support of A & B: " + str(round(support_percent_AandB))
            confidence_list = "Confidence: " + str(round(confidence))
            lift_list = "Lift: " + str(round(lift, 2))
            
            # append rule and its features to the final rule set
            ruleset_apriori.append(rule)
            ruleset_apriori.append(confidence_list)
            ruleset_apriori.append(lift_list)
            ruleset_apriori.append(support_of_A_list)
            ruleset_apriori.append(support_of_B_list)
            ruleset_apriori.append(support_of_AandB_list)
                
    #print(ruleset_apriori)
    return ruleset_apriori


# In[13]:


def print_rules(final_rule_set):
    
    # function to print rules 
    
    position = 0
    # check if ruleset is empty
    if len(final_rule_set) == 0:
        
        print("There are no association rules for this support and confidence.")
        
    else:
        # print generated rules
        print("\n"+'\033[1m' +'RULES GENERATED:'+'\033[0m')
        for i in final_rule_set:
            
            # itemset is in 0th and every 6th location in the list
            if position == 0:
                print("\n\n"+str(i[0]) + "------>" + str(i[1]))
            else:
                print(i,end='  ')
            position = position + 1
            if position == 6:
                position=0
            #print("\n")
        print("\n\n"+'\033[1m' +'Number of rules generated:'+'\033[0m',round(len(final_rule_set)/6))
            


# In[14]:


def display_itemset(itemset):
    
    # function to display itemset based on the size
    item1=[]
    item2=[]
    item3=[]
    item4=[]
    support1=[]
    support2=[]
    support3=[]
    support4=[]
    length_of_itemset=len(itemset)
    i=0
    
    while(i<length_of_itemset):
        
        if len(itemset[i]) == 1:
            item1.append(itemset[i])
            support1.append(itemset[i+1])
            
        elif len(itemset[i]) == 2:
            item2.append(itemset[i])
            support2.append(itemset[i+1])
            
        elif len(itemset[i]) == 3:
            item3.append(itemset[i])
            support3.append(itemset[i+1])
            
        else:
            item4.append(itemset[i])
            support4.append(itemset[i+1])
            
        i=i+2
        
        
    if len(item1) != 0:
        print("\nSingle(1-itemset)")
        df = pd.DataFrame({'Itemset':item1,'support_count':support1})
        display(df)
        
    if len(item2) != 0:
        print("\nPairs(2-itemsets)")
        df = pd.DataFrame({'Itemset':item2,'support_count':support2})
        display(df)
        
    if len(item3) != 0:
        print("\nTriplets(3-itemsets)")
        df = pd.DataFrame({'Itemset':item3,'support_count':support3})
        display(df)
        
    if len(item4) != 0:
        print("\nQuads(4-itemsets)")
        df = pd.DataFrame({'Itemset':item4,'support_count':support4})
        display(df)
        


# In[15]:


# main function
def main():
    
    # function call to all the other functions
    
    dataset,num_of_records = load_data() 

    feature_set = get_feature_set(dataset) 
    
    min_sup,min_conf = get_support_and_confidence()  

    total_frequent_itemsets = []
    
    first_candidate_set = compute_first_set(feature_set)
    print("\n"+'\033[1m' + 'FIRST CANDIDATE SET:' + '\033[0m')
    display_itemset(first_candidate_set)
    
    freq_item_set = generate_itemsets(first_candidate_set, num_of_records, min_sup, feature_set, total_frequent_itemsets)
    
    print("\n"+'\033[1m' + 'LIST OF ITEMSETS:' + '\033[0m')
    display_itemset(total_frequent_itemsets)
    
    sets_for_association_rules = generate_association_rules(total_frequent_itemsets)
    
        
    pruned_rule_set = prune_rules_by_apriori(sets_for_association_rules, feature_set, min_sup, min_conf,num_of_records)
        
    print_rules(pruned_rule_set)


# In[16]:


if __name__ == "__main__":
    
    main()


# In[ ]:




