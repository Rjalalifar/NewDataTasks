import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("sales_data.csv", delimiter=";", encoding="utf-8")

# data preprocessing
data = data[data["Country"] == "United Kingdom"]


data = data[["BillNo", "Itemname"]]

# Create the basket
basket = data.groupby(["BillNo", "Itemname"]).size().unstack(fill_value=0)
basket[basket > 0] = 1

# Apriori Algorithm 
def Support(itemset, basket):
    return (basket[list(itemset)].sum(axis=1) == len(itemset)).mean()

def Confidence(rule, basket):
    antecedent = rule[0]
    consequent = rule[1]
    support_antecedent = Support(antecedent, basket)
    support_rule = Support(antecedent | consequent, basket)
    return support_rule / support_antecedent

def Lift(rule, basket):
    antecedent = rule[0]
    consequent = rule[1]
    support_antecedent = Support(antecedent, basket)
    support_consequent = Support(consequent, basket)
    support_rule = Support(antecedent | consequent, basket)
    return support_rule / (support_antecedent * support_consequent)

# Find frequent itemsets
min_support = 0.02
frequent_itemsets = []
for column in basket.columns:
    support = Support([column], basket)
    if support >= min_support:
        frequent_itemsets.append([column])

# Find association rules
min_lift = 1.0
association_rules = []
for i in range(len(frequent_itemsets)):
    for j in range(i + 1, len(frequent_itemsets)):
        antecedent = set(frequent_itemsets[i])
        consequent = set(frequent_itemsets[j])
        rule = (antecedent, consequent)
        lift = Lift(rule, basket)
        if lift >= min_lift:
            association_rules.append((antecedent, consequent, lift))

# Display association rules
for rule in association_rules:
    antecedent, consequent, lift = rule
    support = Support(antecedent | consequent, basket)
    confidence = Confidence((antecedent, consequent), basket)
    print(f"Antecedent: {antecedent}, Consequent: {consequent}, Support: {support}, Confidence: {confidence}, Lift: {lift}")

# Create DataFrame to association rules
df_rules = pd.DataFrame(association_rules, columns=["Antecedent", "Consequent", "Lift"])

df_rules = df_rules.sort_values(by="Lift", ascending=False)

top_n = 10  
df_top_n = df_rules.head(top_n)


plt.figure(figsize=(10, 6))
plt.barh(df_top_n["Antecedent"].apply(str) + " -> " + df_top_n["Consequent"].apply(str), df_top_n["Lift"])
plt.xlabel("Lift")
plt.ylabel("Association Rule")
plt.title("Top 10 Association Ruls")
plt.gca().invert_yaxis() 
plt.show()