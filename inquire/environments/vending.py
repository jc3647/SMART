import numpy as np

class FoodItem:
    def __init__(self, itemName, ifSolid, softness, price, salt_content, healthiness_index, sugar_content, protein, calories):
        self.itemName = itemName
        self.ifSolid = ifSolid
        self.softness = softness
        self.price = price
        self.salt_content = salt_content
        self.healthiness_index = healthiness_index
        self.sugar_content = sugar_content
        self.protein = protein
        self.calories = calories
        
    def get_features(self):
        return np.array([self.ifSolid, self.softness, self.price, self.salt_content, self.healthiness_index, 
                         self.sugar_content, self.protein, self.calories])

class VendingMachine:
    def __init__(self, food_items, K):
        self.food_items = food_items
        self.K = K  # Number of prototypes
        self.prototypes = None
        self.Q_k_matrices = None
        self.initialize_prototypes()

    def initialize_prototypes(self):
        L = len(self.food_items)
        self.prototypes = np.random.choice(self.food_items, self.K, replace=False)  # Randomly select K prototypes
        
        # Initialize Q_k matrices with uniform preferences
        self.Q_k_matrices = np.full((self.K, L, L), 0.5)  # 0.5 indicates no preference between any two items

    def update_preferences(self, chosen_index, compared_index, outcome):
        # print("outcome: ", outcome)
        # Direct adjustment based on outcome
        for k in range(self.K):
            self.Q_k_matrices[k][chosen_index][compared_index] = min(1, self.Q_k_matrices[k][chosen_index][compared_index] + outcome)
            self.Q_k_matrices[k][compared_index][chosen_index] = max(0, self.Q_k_matrices[k][compared_index][chosen_index] - outcome)
    # def provide_feedback(self, user_preference):
    #     # Compute scores for all items based on distance to user_preference
    #     scores = self.compute_scores(user_preference)
        
    #     # Update Q_k_matrices based on scores
    #     for i, score_i in enumerate(scores):
    #         for j, score_j in enumerate(scores):
    #             if i != j:
    #                 outcome = score_i - score_j
    #                 self.update_preferences(i, j, outcome)

    def provide_feedback(self, user_preference):
        scores = self.compute_scores(user_preference)
        L = len(self.food_items)
        
        # Compute assignment probabilities for each food item
        assignment_probabilities = np.zeros((L, self.K))
        for i, food_item in enumerate(self.food_items):
            assignment_probabilities[i] = self.compute_assignment_probability(food_item.get_features(), sigma_p=1)  # sigma_p needs to be defined based on your system
            # print("-----------------")
            # print("food item: ", food_item.itemName)
            # print("assignment_probabilities: ", assignment_probabilities[i])
        # Update Q_k_matrices based on both scores and assignment probabilities
        for i in range(L):
            for j in range(L):
                if i != j:
                    # Calculate influence based on both scores and assignment probabilities
                    score_difference = scores[i] - scores[j]
                    # print("score_difference: ", score_difference)
                    # probability_influence = np.sum(assignment_probabilities[i] - assignment_probabilities[j])
                    outcome = score_difference # * probability_influence
                    
                    self.update_preferences(i, j, outcome)

    def compute_scores(self, user_preference):
        L = len(self.food_items)
        scores = np.zeros(L)
        max_distance = 0
        
        
        # Calculate distances and find max for normalization
        for i, food_item in enumerate(self.food_items):
            distance = np.linalg.norm(user_preference - food_item.get_features())
            scores[i] = distance
            if distance > max_distance:
                max_distance = distance
        
        # Normalize distances to scores (inverse to make closer items have higher scores)
        for i in range(L):
            scores[i] = 1 - (scores[i] / max_distance)
            
        return scores

    def compute_assignment_probability(self, xn, sigma_p):
        probabilities = np.zeros(self.K)
        for k, mk in enumerate(self.prototypes):
            # Calculate the squared Euclidean distance between xn and mk (prototype)
            distance_squared = np.sum((xn - mk.get_features())**2)
            probabilities[k] = np.exp(-distance_squared / (2 * sigma_p**2))
        
        probabilities /= np.sum(probabilities)  # Normalize the probabilities
        return probabilities
    
    def compute_posterior_probability(self, Yn, sigma_y, probabilities):
        # calculate the Gaussian error term for prototype k
        enk = np.zeros(self.K)
        
        for k, Qk in enumerate(self.Q_k_matrices):
            distances_squared = np.sum((Yn - np.array([food.get_features() for food in self.food_items]))**2, axis=1)
            print("distances_squared: ", distances_squared)
            #enk[k] = np.exp(-np.linalg.norm((Yn - Qk)**2) / (2 * sigma_y**2))
            enk[k] = np.exp(-np.linalg.norm(distances_squared * Qk) / (2 * sigma_y**2))
            
        
        # Calculate the posterior probability as a weighted sum of the error terms
        posterior_prob = np.sum(probabilities * enk)
        print("posterior prob: ", posterior_prob)
        return posterior_prob

    def test(self):
        for p in self.prototypes:
            print(p.itemName)
        # print("Q_k_matrices: ", self.Q_k_matrices)

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

vending_machine = VendingMachine(food_items, 20)
vending_machine.provide_feedback(food_items[0].get_features())
print("Q_k_matrices: ", vending_machine.Q_k_matrices[0][0])
# for i in range(20):
#     print("vending machine test: ", vending_machine.Q_k_matrices[i])
# assignment_probability = vending_machine.compute_assignment_probability(food_items[1].get_features(), 1)
# print("assignment_probability: ", assignment_probability)
# vending_machine.compute_posterior_probability(food_items[0].get_features(), 1, assignment_probability)
# vending_machine.test()


