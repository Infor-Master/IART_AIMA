# (dataframe) pandas <-> (bunch) sklearn
# dataset -> csv / arff

from munch import munchify
from pathlib import Path
from io import StringIO
import arff
import numpy as np
import pandas as pd
import sklearn as sk
import pydot


base_path = Path(__file__).parent
file_path = (base_path / "../../../aima-data/restaurant.csv").resolve()

# Import .csv as pandas Dataframe
columns = ["Alternate", "Bar", "Fri/Sat", "Hungry", "Patrons", "Price", "Raining", "Reservation", "Type", "WaitEstimate", "WillWait"]
df = pd.read_csv(file_path, names=columns)

# Dataframe to Bunch
data_dict = df.to_dict()
data_bunch = munchify(data_dict) #bunch is just a fancy dictionary

# Bunch to Dataframe
data_df = pd.DataFrame.from_dict(data_bunch)

# Export to .arff
dump_arff_path = (base_path / "restaurant.arff").resolve()
arff.dump(dump_arff_path, 
        df.values,  
        names=df.columns)

# Export to .csv
dump_csv_path = (base_path / "restaurant.csv").resolve()
df.to_csv(dump_csv_path)

print(df.head())

### decision tree ###

from sklearn import tree

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[str(target_column + "_int")] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

labels = []
for col in columns:
    df, target = encode_target(df, col)
    labels.append([col, target])

features = ["Alternate_int", "Bar_int", "Fri/Sat_int", "Hungry_int", "Patrons_int", "Price_int", "Raining_int", "Reservation_int", "Type_int", "WaitEstimate_int"]

y = df["WillWait_int"]
X = df[features]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# save tree as .dot and as .png

sk.tree.export_graphviz(clf,
            feature_names = features,
            out_file='./IART/exercícios/FichaML/dtree.dot')

dotfile = StringIO()
sk.tree.export_graphviz(clf, 
            feature_names = features,
            out_file=dotfile)
(graph,) = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("./IART/exercícios/FichaML/dtree.png")

print(labels)
