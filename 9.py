from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample dataset
dataset = [['Milk', 'Eggs', 'Bread'],
           ['Milk', 'Cookies'],
           ['Bread', 'Eggs', 'Cookies'],
           ['Milk', 'Bread', 'Eggs', 'Cookies'],
           ['Bread', 'Cookies']]

# Convert the dataset to the required format
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display frequent itemsets and rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
