# disturbance_uncertainty

### _How to set up JSON file to run experiments.  Claude wrote a basic template_

Written to experiments/configs/baseline.json. Here's what each top-level block maps to:

JSON key	Used by
landscape_stack_config	LandscapeStack.from_config() / LandscapeStackCollection.from_config()
training_experiment_config	LandscapeStackCollection.from_config() for the training collection
training_model_config	buildModel() arguments
training_interp_prob_model_config	buildInterpreterProbModel() arguments
training_interp_config	InterpreterAgent(**...) for model building
binary_classifier_config	buildBinaryClassifier() arguments
validation_experiment_config	LandscapeStackCollection.from_config() for the validation collection
validation_interp_config	InterpreterAgent(**...) for validation
evaluate_config	evaluate_validation_collection() arguments
stack_plots_config	save_all_stack_plots() arguments
A few things to note:

null = Python None, true/false = Python booleans
The interpreter types block in landscape_stack_config.interpreter.config is currently set to match the global defaults — once you want type1/type2 to behave differently, that's where to edit
tif_path in landscape_stack_config is only used as a fallback — patch_dir in the experiment configs drives raster selection when running a collection
