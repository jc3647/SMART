
import numpy as np
from inquire.utils.features import FoodItem

food_items = [
    FoodItem("Regular Potato Chips (Lays)", 1, 5.8, 2.00, 170, 52, 1, 2, 160),
    FoodItem("Barbeque Chips", 1, 6, 2.50, 150, 53, 2, 2, 160),
    FoodItem("Sour Cream Chips", 1, 5.6, 2.50, 200, 53, 2, 2, 160),
    FoodItem("Flaming Hot Cheetos", 1, 6.2, 3.00, 250, 21, 0, 1, 170),
    FoodItem("Coca-Cola", 0, 1, 1.50, 45, 1, 39, 0, 140),
    FoodItem("Coffee", 0, 1, 3.00, 120, 1, 12, 8, 140),
    FoodItem("Orange Juice", 0, 1, 2.00, 3, 79, 28, 3, 153),
    FoodItem("Water", 0, 1, 1.00, 0, 100, 0, 0, 0),
    FoodItem("Hershey's Chocolate", 1, 4.2, 1.50, 30, 14, 20, 4, 210),
    FoodItem("Reese's Peanut Butter Cups", 1, 3.2, 1.50, 150, 28, 22, 5, 210),
    FoodItem("Skittles", 1, 6.2, 1.50, 20, 1, 45, 0, 250),
    FoodItem("Kit Kat", 1, 4.4, 1.50, 30, 6, 22, 3, 210),
    FoodItem("Apple", 1, 6.4, 0.50, 2, 92, 19, 1, 95),
    FoodItem("Banana", 1, 2.4, 0.50, 0, 79, 17, 1, 120),
    FoodItem("Pear", 1, 5.8, 0.75, 0, 82, 16, 1, 100),
    FoodItem("Orange", 1, 3.4, 0.75, 0, 100, 14, 1, 70),
    FoodItem("Popcorn", 1, 5, 1.50, 820, 61, 0, 2, 425),
    FoodItem("Ramen", 1, 2.5, 0.50, 1590, 1, 0, 9, 380),
    FoodItem("Mac and Cheese", 1, 2.7, 1.50, 490, 11, 5, 7, 220),
    FoodItem("Chicken Noodle Soup", 1, 2.2, 3.00, 2225, 21, 1, 8, 150)
]


class VendingMachine():
    def __init__(self):
        self.food_items = food_items
        self.food_items_positions = np.arange(len(self.food_items))

    def pick_random_food(self):
        return np.random.choice(self.food_items).itemName
    
test = VendingMachine()
for i in range(10):
    print(test.pick_random_food())
    

        

# class BayesianIRL:
#     def __init__(self, num_reward_functions, prior_mean=0, prior_variance=1):
#         self.num_reward_functions = num_reward_functions
#         self.prior_mean = prior_mean
#         self.prior_variance = prior_variance
#         self.reward_functions = np.random.normal(self.prior_mean, np.sqrt(self.prior_variance), size=(self.num_reward_functions,))
#         self.posterior_distribution = np.ones(self.num_reward_functions) / self.num_reward_functions

#     def print_all(self):
#         print("num_reward_functions: ", self.num_reward_functions)
#         print("prior_mean: ", self.prior_mean)
#         print("prior_variance: ", self.prior_variance)
#         print("reward_functions: ", self.reward_functions)
#         print("posterior_distribution: ", self.posterior_distribution)

#     def update_prior(self, new_prior_mean, new_prior_variance):
#         self.prior_mean = new_prior_mean
#         self.prior_variance = new_prior_variance
#         self.reward_functions = np.random.normal(self.prior_mean, np.sqrt(self.prior_variance), size=(self.num_reward_functions,))
#         self.posterior_distribution = np.ones(self.num_reward_functions) / self.num_reward_functions
        



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
# num_reward_functions = 10
# prior_mean = 0
# prior_variance = 1
# bayesian_irl = BayesianIRL(num_reward_functions, prior_mean, prior_variance)
# bayesian_irl.print_all()





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
