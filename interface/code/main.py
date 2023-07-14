from components import *
from analysis import *

"""
create all component objects for the analysis
"""
new_dataset = Dataset().create()
new_database = Database().create()
new_leakage_model = LeakageModel().create()
new_neural_network = NeuralNetwork().create()
new_metric = ScaMetric().create()
new_plot = Plot().create()
new_profiling = Profiling().create()
new_callback = Callback().create()

"""
create analysis and add all components to the analysis
"""
my_analysis = Analysis()
my_analysis.add_component(new_dataset)
my_analysis.add_component(new_database)
my_analysis.add_component(new_leakage_model)
my_analysis.add_component(new_neural_network)
my_analysis.add_component(new_metric)
my_analysis.add_component(new_callback)
my_analysis.add_component(new_profiling)
my_analysis.add_component(new_plot)

"""
run analysis
1. Read new_dataset
2. Create new_leakage_model
3. Define new_neural_network
4. Instantiate new_metric
5. Instantiate new_callback
6. Implement new_profiling
7. Plot new_plot
8. Store results in new_database
"""
my_analysis.run()
