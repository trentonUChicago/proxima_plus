import numpy as np
import time
from sklearn.linear_model import LinearRegression


class ControlWrapper:
    """Control Model that manages when to use target or surrogate model"""

    def __init__(self, target_function, surrogate, acceptable_error, initial_surrogate_data = 20, 
                 initial_error_data = 20, retrain_interval=10, error_prediction_type=None, 
                 sliding_window_size = 10):
        """
        target_function is the actual target model
        surrogate_class is assumed to be a class that an instance can be created with
        """

        # Set Target and Surrogate Function
        self.target_function = target_function
        self.surrogate = surrogate # We assume that this is an ensemble model

        # Set Controller Variables
        self.acceptable_error = acceptable_error # Set error threshold for surrogate error
        self.surrogate_initialized = False
        self.initial_surrogate_data = initial_surrogate_data
        self.initial_error_data = initial_error_data
        self.retrain_interval = retrain_interval
        
        # History
        self.num_runs = 0
        self.was_target_ran = False
        self.results = {"X": [], "y_target": [], "y_prediction": [], "epistemic_uncertainty": [], "aleatory_uncertainty": [], "coefficient_of_determination": []}
        self.training_data = []
        self.retraining_times = []

        # Variables for the error prediction
        self.error_model = LinearRegression()
        self.error_prediction_type = error_prediction_type
        self.error_training_size = 0
        self.sliding_window_size = sliding_window_size
        self.error_sliding_window = {"surrogate_error": [], "epistemic_uncertainty": [], "aleatory_uncertainty": []} # Temp window where values are added and removed
        self.error_prediction_vals = {"surrogate_error": [], "epistemic_uncertainty": [], "aleatory_uncertainty": []}
        self.r_squared = 0
        
    def __call__(self, x):
        """Decide whether to run surrogate or target model and update params"""

        self.num_runs += 1
        
        # Get prediction from surrogate along with it's measures of uncertainty
        surrogate_value, epistemic_uncertainty, aleatory_uncertainty = self.run_surrogate(x)

        # If there is a prediction value, use the uncertainties to decide if we should use surrogate
        if surrogate_value is not None:
            error_pred = self.get_error_prediction(epistemic_uncertainty, aleatory_uncertainty)
            # Check error prediction with threshold
            use_surrogate = error_pred < self.acceptable_error
        else:
            use_surrogate = False

        # Run Target Model
        target_value = None
        if not(use_surrogate):
            target_value = self.run_target_model(x)    

        # Update results history and data
        self.record_info(x, target_value, surrogate_value, epistemic_uncertainty, aleatory_uncertainty)
        
        return (surrogate_value if use_surrogate else target_value), use_surrogate

    def run_surrogate(self, x):
        """Runs surrogate model as an ensemble and return means and variances"""

        # If there is not enough data to initialize the surrogate model, then just run target model
        if (len(self.results["X"]) < self.initial_surrogate_data) or (self.acceptable_error <= 0):
            return None, None, None
    
        # See if we need to train/retrain surrogate model
        if not(self.surrogate_initialized):
            self.retrain_surrogate()
            self.surrogate_initialized = True
            self.was_target_ran = False
        elif (self.num_runs % self.retrain_interval == 0) and self.was_target_ran:
            self.retrain_surrogate()
            self.was_target_ran = False

        # Run surrogate ensemble model
        surrogate_value, epistemic_uncertainty, aleatory_uncertainty = self.surrogate.predict([x], return_predictive_error=True)

        return surrogate_value[0], epistemic_uncertainty[0], aleatory_uncertainty[0]

    def retrain_surrogate(self):
        """Retrains each surrogate from stratch using previous results"""
        start_time = time.time()
        # Get X and y for training from previous target model results
        X, y_target = self.get_surrogate_training_data()
        self.surrogate.fit(X, y_target)
        end_time = time.time()
        # Save surrogate retraining time
        self.retraining_times.append({"run_num": self.num_runs, "data_length": len(X), "run_time": end_time - start_time})
    
    def get_surrogate_training_data(self):
        """Get valid training data"""
        X = []
        y_target = []
        for i in range(len(self.results['X'])):
            # Check if y_target is available
            if self.results["y_target"][i] is not None:
                X.append(self.results['X'][i])
                y_target.append(self.results["y_target"][i])
        
        return X, y_target

    def run_target_model(self, x):
        """Run the high fidelity target model"""
        self.was_target_ran = True
        y = self.target_function(x)
        return y

    def get_error_prediction(self, epistemic_uncertainty, aleatory_uncertainty):
        """Predict the surrogate error given aleatory and epistemic uncertainty"""
        if len(self.error_prediction_vals["surrogate_error"]) < self.initial_error_data:
            return np.inf
        
        # Check if there is new data to train error predictor
        if self.error_training_size < len(self.error_prediction_vals["surrogate_error"]):
            self.error_training_size = len(self.error_prediction_vals["surrogate_error"])

            # Get values
            surrogate_errors = self.error_prediction_vals["surrogate_error"]
            epistemic_errors = self.error_prediction_vals["epistemic_uncertainty"]
            aleatory_errors = self.error_prediction_vals["aleatory_uncertainty"]

            # Pair uncertainties together
            uncertainties = np.array([list(pair) for pair in zip(epistemic_errors, aleatory_errors)])

            # Fit model
            self.error_model.fit(np.log10(uncertainties), surrogate_errors)

            # Save coefficient of determination
            self.r_squared = self.error_model.score(np.log10(uncertainties), surrogate_errors)

        surrogate_error_pred = self.error_model.predict([np.log10([epistemic_uncertainty, aleatory_uncertainty])])[0]

        return surrogate_error_pred

    def record_info(self, x, target_value, surrogate_value=None, epistemic=None, aleatory=None):
        """Simply saves information about the values and uncertainties over time"""
        self.results["X"].append(x)
        self.results["y_target"].append(target_value)
        self.results["y_prediction"].append(surrogate_value)
        self.results["epistemic_uncertainty"].append(epistemic)
        self.results["aleatory_uncertainty"].append(aleatory)
        self.results["coefficient_of_determination"].append(self.r_squared)

        # If we can find surrogate error, use to update sliding window for error prediction
        if (target_value is not None) and (surrogate_value is not None):
            absolute_error = np.abs(target_value - surrogate_value)
            self.update_error_sliding_window(absolute_error, epistemic, aleatory)

    def update_error_sliding_window(self, surrogate_error, epistemic, aleatory):
        # Add new values to sliding window
        self.error_sliding_window["surrogate_error"].append(surrogate_error)
        self.error_sliding_window["epistemic_uncertainty"].append(epistemic)
        self.error_sliding_window["aleatory_uncertainty"].append(aleatory)

        # Check if the sliding window is too big and needs values removed
        if len(self.error_sliding_window["surrogate_error"]) > self.sliding_window_size:
            self.error_sliding_window["surrogate_error"] = self.error_sliding_window["surrogate_error"][-self.sliding_window_size:]
            self.error_sliding_window["epistemic_uncertainty"] = self.error_sliding_window["epistemic_uncertainty"][-self.sliding_window_size:]
            self.error_sliding_window["aleatory_uncertainty"] = self.error_sliding_window["aleatory_uncertainty"][-self.sliding_window_size:]
        
        # Add new sliding window value if possible 
        if len(self.error_sliding_window["surrogate_error"]) == self.sliding_window_size:
            self.error_prediction_vals["epistemic_uncertainty"].append(np.mean(self.error_sliding_window["epistemic_uncertainty"]))
            self.error_prediction_vals["aleatory_uncertainty"].append(np.mean(self.error_sliding_window["aleatory_uncertainty"]))

            # Check error prediction type
            if self.error_prediction_type == "max":
                self.error_prediction_vals["surrogate_error"].append(np.max(self.error_sliding_window["surrogate_error"]))
            elif isinstance(self.error_prediction_type, int):
                self.error_prediction_vals["surrogate_error"].append(np.percentile(self.error_sliding_window["surrogate_error"], self.error_prediction_type))
            else:
                self.error_prediction_vals["surrogate_error"].append(np.mean(self.error_sliding_window["surrogate_error"]))
        