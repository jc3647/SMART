
import numpy as np

class BayesianIRL:
    def __init__(self, num_reward_functions, prior_mean=0, prior_variance=1):
        self.num_reward_functions = num_reward_functions
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.reward_functions = np.random.normal(self.prior_mean, np.sqrt(self.prior_variance), size=(self.num_reward_functions,))
        self.posterior_distribution = np.ones(self.num_reward_functions) / self.num_reward_functions

    def print_all(self):
        print("num_reward_functions: ", self.num_reward_functions)
        print("prior_mean: ", self.prior_mean)
        print("prior_variance: ", self.prior_variance)
        print("reward_functions: ", self.reward_functions)
        print("posterior_distribution: ", self.posterior_distribution)

    def update_prior(self, new_prior_mean, new_prior_variance):
        self.prior_mean = new_prior_mean
        self.prior_variance = new_prior_variance
        self.reward_functions = np.random.normal(self.prior_mean, np.sqrt(self.prior_variance), size=(self.num_reward_functions,))
        self.posterior_distribution = np.ones(self.num_reward_functions) / self.num_reward_functions
        



    # def update_prior(self, new_prior_mean, new_prior_variance):
    #     self.prior_mean = new_prior_mean
    #     self.prior_variance = new_prior_variance

    # def update_posterior(self, demonstration, foods):
    #     likelihoods = self.calculate_likelihoods(demonstration, foods)
    #     self.posterior_distribution *= likelihoods
    #     self.posterior_distribution /= np.sum(self.posterior_distribution)

    # def calculate_likelihoods(self, demonstration, foods, weight_vector):
    #     demonstration_features = demonstration["features"]
    #     demonstration_score = demonstration["score"]
    #     likelihoods = []
    #     for food in foods:
    #         food_features = food["features"]
    #         feature_distances = np.abs(demonstration_features - food_features)
    #         weighted_distances = np.dot(feature_distances, weight_vector)
    #         likelihood = np.exp(-weighted_distances) if demonstration_score == food["score"] else 0
    #         likelihoods.append(likelihood)
    #     return np.array(likelihoods)

# Example Initialization
num_reward_functions = 10
prior_mean = 0
prior_variance = 1
bayesian_irl = BayesianIRL(num_reward_functions, prior_mean, prior_variance)
bayesian_irl.print_all()





# # Example demonstration and foods
# demonstration = {"features": np.array([0.8, 0.2, 0.5, 0.7, 0.6, 0.3, 0.4, 0.9]), "score": 4.5}
# foods = [
#     {"features": np.array([0.7, 0.3, 0.4, 0.6, 0.5, 0.2, 0.3, 0.8]), "score": 4.5},
#     {"features": np.array([0.6, 0.4, 0.6, 0.5, 0.4, 0.1, 0.2, 0.7]), "score": 3.8},
#     # Add more food items here
# ]

# # Example weight vector
# weight_vector = np.array([0.2, 0.3, 0.1, 0.2, 0.1, 0.05, 0.05, 0.1])

# # Update posterior distribution
# bayesian_irl.update_posterior(demonstration, foods, weight_vector)
