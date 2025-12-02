import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def get_distance(model, training_data, x, k=1, n_jobs=None, metric='mahalanobis'):
    # Get distance
    if len(training_data) == 0 or not(model.fitted_):
        return 0

    training_data = model.data_pipeline.transform(training_data)
    x = model.data_pipeline.transform([x])

    nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, metric=metric).fit(training_data)
    dists, _ = nn.kneighbors(x)
    return np.min(dists, axis=1)


class ControlWrapper:
    """Control Model that manages when to use target or surrogate model"""

    def __init__(self, target_function, surrogate, acceptable_error, initial_surrogate_data = 20, 
                 initial_error_data = 20, retrain_interval=None, error_prediction_type=None, 
                 sliding_window_size = 10, prediction_window_size = 0, epsilon = 0.1,
                 threshold_type = "error_pred_fixed_threshold"):
        """
        target_function is the actual target model
        surrogate is assumed to be an ensemble model
        """

        # Set Target and Surrogate Function
        self.target_function = target_function
        self.surrogate = surrogate # We assume that this is an ensemble model

        # Set Controller Variables
        self.threshold_window = 100
        self.acceptable_error = acceptable_error # Set error threshold for surrogate error
        self.disable_surrogate = acceptable_error == 0 
        self.surrogate_initialized = False
        self.initial_surrogate_data = initial_surrogate_data
        self.initial_error_data = initial_error_data
        self.retrain_interval = retrain_interval
        self.epsilon = epsilon # Chance of running target model anyways
        self.threshold_type = threshold_type
        if self.threshold_type == "proxima":
            self.threshold = None

        
        # History
        self.num_runs = 0
        self.was_target_ran = False
        self.results = {"X": [], "y_target": [], "y_prediction": [], "surrogate_error": [], "epistemic_uncertainty": [], "aleatory_uncertainty": []}
        if self.threshold_type == "error_pred_fixed_threshold":
            self.results["coefficient_of_determination"] = []
            self.results["acceptable_surrogate_error"] = []
            self.results["surrogate_error_prediction"] = []
        elif self.threshold_type == "proxima":
            self.results["distance"] = []
            self.results["threshold"] = []
        self.retraining_times = []
        self.last_retrain_data_size = 0

        # Variables for the error prediction
        self.error_model = LinearRegression()
        self.error_prediction_type = error_prediction_type
        self.error_training_size = 0
        self.prediction_window_size = prediction_window_size # Number of the most recent values used to train error predictor
        self.sliding_window_size = sliding_window_size
        # Temp window where values are added and removed
        self.error_sliding_window = {"surrogate_error": [], "epistemic_uncertainty": [], "aleatory_uncertainty": [], "distance": []}
        self.error_prediction_vals = {"surrogate_error": [], "epistemic_uncertainty": [], "aleatory_uncertainty": [], "distance": []}
        self.r_squared = 0
        
    def __call__(self, x):
        """Decide whether to run surrogate or target model and update params"""

        self.num_runs += 1
        
        # Get prediction from surrogate along with it's measures of uncertainty
        surrogate_value, epistemic_uncertainty, aleatory_uncertainty = self.run_surrogate(x)

        # If there is a prediction value, use the uncertainties to decide if we should use surrogate
        if surrogate_value is not None:
            if self.threshold_type == "error_pred_fixed_threshold":
                # Get a predicted surrogate error
                uncertainty = self.get_error_prediction(epistemic_uncertainty, aleatory_uncertainty)
                # Check error prediction with threshold
                use_surrogate = uncertainty < self.acceptable_error
            elif self.threshold_type == "proxima":
                # Get minimum distance
                uncertainty = get_distance(self.surrogate, self.get_surrogate_training_data()[0], x)
                # Check if minimum distance is 
                use_surrogate = self.threshold is not None and uncertainty < self.threshold


        else:
            uncertainty = None
            use_surrogate = False

        # Run Target Model
        target_value = None
        if not(use_surrogate) or (np.random.random() < self.epsilon):
            target_value = self.run_target_model(x)
            use_surrogate = False

        # Update results history and data
        self.record_info(x, target_value, surrogate_value, epistemic_uncertainty, aleatory_uncertainty, uncertainty)
        return (surrogate_value if use_surrogate else target_value), use_surrogate

    def run_surrogate(self, x):
        """Runs surrogate model as an ensemble and return means and variances"""

        # If there is not enough data to initialize the surrogate model, then just run target model
        if (len(self.results["X"]) < self.initial_surrogate_data) or self.disable_surrogate:
            return None, None, None
    
        # See if we need to train/retrain surrogate model
        if not(self.surrogate_initialized):
            self.retrain_surrogate()
            self.surrogate_initialized = True
            self.was_target_ran = False
        elif self.retrain_interval is None:
            X, _ = self.get_surrogate_training_data()
            if self.was_target_ran and (((self.last_retrain_data_size < 100) and (len(X) % 10 == 0)) or \
               ((self.last_retrain_data_size >= 100) and (self.last_retrain_data_size * 1.1 < len(X)))):
                self.retrain_surrogate()
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
        print("In retrain_surrogate, X type:", type(X))
        self.surrogate.fit(X, y_target)
        end_time = time.time()
        # Save surrogate retraining time
        self.last_retrain_data_size = len(X)
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
            surrogate_errors = self.error_prediction_vals["surrogate_error"][-self.prediction_window_size:]
            epistemic_errors = self.error_prediction_vals["epistemic_uncertainty"][-self.prediction_window_size:]
            aleatory_errors = self.error_prediction_vals["aleatory_uncertainty"][-self.prediction_window_size:]

            # Pair uncertainties together
            uncertainties = np.array([list(pair) for pair in zip(epistemic_errors, aleatory_errors)])

            # Fit model
            self.error_model.fit(np.log10(uncertainties), surrogate_errors)

            # Save coefficient of determination
            self.r_squared = self.error_model.score(np.log10(uncertainties), surrogate_errors)

        surrogate_error_pred = self.error_model.predict([np.log10([epistemic_uncertainty, aleatory_uncertainty])])[0]

        return surrogate_error_pred

    def record_info(self, x, target_value, surrogate_value=None, epistemic=None, aleatory=None, uncertainty=None):
        """Simply saves information about the values and uncertainties over time"""
        self.results["X"].append(x)
        self.results["y_target"].append(target_value)
        self.results["y_prediction"].append(surrogate_value)
        self.results["epistemic_uncertainty"].append(epistemic)
        self.results["aleatory_uncertainty"].append(aleatory)

        if self.threshold_type == "error_pred_fixed_threshold":
            self.results["coefficient_of_determination"].append(self.r_squared)
            self.results["acceptable_surrogate_error"].append(self.acceptable_error)
            self.results["surrogate_error_prediction"].append(uncertainty)
        elif self.threshold_type == "proxima":
            self.results["distance"].append(uncertainty)
            self.results["threshold"].append(self.threshold)

        # If we can find surrogate error, use to update sliding window for error prediction
        if (target_value is not None) and (surrogate_value is not None):
            absolute_error = np.abs(target_value - surrogate_value)
            self.update_error_sliding_window(absolute_error, epistemic, aleatory, uncertainty)
        else:
            self.results["surrogate_error"].append(None)

    def update_error_sliding_window(self, surrogate_error, epistemic, aleatory, distance=None):
        self.results["surrogate_error"].append(surrogate_error)
        # Add new values to sliding window
        self.error_sliding_window["surrogate_error"].append(surrogate_error)
        self.error_sliding_window["epistemic_uncertainty"].append(epistemic)
        self.error_sliding_window["aleatory_uncertainty"].append(aleatory)
        if self.threshold_type == "proxima":
            self.error_sliding_window["distance"].append(distance)

        # Check if the sliding window is too big and needs values removed
        if len(self.error_sliding_window["surrogate_error"]) > self.sliding_window_size:
            self.error_sliding_window["surrogate_error"] = self.error_sliding_window["surrogate_error"][-self.sliding_window_size:]
            self.error_sliding_window["epistemic_uncertainty"] = self.error_sliding_window["epistemic_uncertainty"][-self.sliding_window_size:]
            self.error_sliding_window["aleatory_uncertainty"] = self.error_sliding_window["aleatory_uncertainty"][-self.sliding_window_size:]
            if self.threshold_type == "proxima":
                self.error_sliding_window["distance"] = self.error_sliding_window["distance"][-self.sliding_window_size:]
        
        # Add new sliding window value if possible 
        if len(self.error_sliding_window["surrogate_error"]) == self.sliding_window_size:
            self.error_prediction_vals["epistemic_uncertainty"].append(np.mean(self.error_sliding_window["epistemic_uncertainty"]))
            self.error_prediction_vals["aleatory_uncertainty"].append(np.mean(self.error_sliding_window["aleatory_uncertainty"]))
            if self.threshold_type == "proxima":
                self.error_prediction_vals["distance"].append(np.mean(self.error_sliding_window["distances"]))

            # Check error prediction type
            if self.error_prediction_type == "max":
                self.error_prediction_vals["surrogate_error"].append(np.max(self.error_sliding_window["surrogate_error"]))
            elif isinstance(self.error_prediction_type, int):
                self.error_prediction_vals["surrogate_error"].append(np.percentile(self.error_sliding_window["surrogate_error"], self.error_prediction_type))
            else:
                self.error_prediction_vals["surrogate_error"].append(np.mean(self.error_sliding_window["surrogate_error"]))
        
    # def update_threshold(self, kT, final_error_bound, max_final_value_change):
    #     if max_final_value_change > 0:
    #         self.acceptable_error = kT * np.log(1 + final_error_bound/max_final_value_change)

    def update_threshold(self):
        # Get surrogate_errors
        surrogate_err = []
        for i in range(len(self.results['surrogate_error'])):
            if self.results["surrogate_error"][i] is not None:
                surrogate_err.append(self.results["surrogate_error"][i])

        # If no surrogate errors are available, return
        if len(surrogate_err) == 0:
            return

        self.update_alpha()
        
        current_err = np.mean(surrogate_err[-self.threshold_window:])

        if self.threshold == None:
            # Following Eq. 1 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370
            self.threshold = current_err / self.alpha
        else:
            # Update according to Eq. 3 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370
            self.threshold -= (current_err - self.acceptable_error) / self.alpha
            self.threshold = max(self.threshold, 0)  # Keep it at least zero

    def update_alpha(self):
        self.alpha = stats.linregress(self.error_prediction_vals["distance"][-self.threshold_window:], 
                                      self.error_prediction_vals["surrogate_error"][-self.threshold_window:]).slope
        self.alpha = max(self.alpha, 1e-6)
