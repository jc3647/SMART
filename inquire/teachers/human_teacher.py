from inquire.teachers.teacher import Teacher
from inquire.utils.datatypes import Choice, Query, Feedback, Modality, Trajectory
from typing import Union
from inquire.environments.environment import CachedTask, Task
from inquire.environments.vm import FoodItem
import numpy as np

class HumanTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha
    
    def __init__(self, N, vending_machine, display_interactions: bool = False) -> None:
        super().__init__()
        self._alpha = 0.75
        self._N = N
        self._display_interactions = display_interactions
        # print("vending machine: ", vending_machine.food_items)
        self.vending_machine_items = vending_machine.food_items
    
    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        if q.query_type is Modality.DEMONSTRATION:
            f = self.demonstration(q, task)
        elif q.query_type is Modality.PREFERENCE:
            f = self.preference(q, task)
        elif q.query_type is Modality.BINARY:
            pass
        return f
        
    def demonstration(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        print("Please provide your favorite food.")
        favorite_food = input()
        try:
            demo = self.vending_machine_items[favorite_food].get_features()
            print("these are the features of: ", favorite_food, " ", demo, ", normalized!")
            traj = Trajectory(states=np.expand_dims(demo, axis=0), actions=None, phi=demo)
            print("You have chosen: ", favorite_food)
        except:
            print("I'm sorry, I don't have that item in my vending machine.")
            return

        vending_machine_trajectories = [Trajectory(
            states=np.expand_dims(food.get_features(), axis=0),
            actions=None,
            phi=food.get_features()
        ) for food in self.vending_machine_items.values()]

        f = Feedback(
            Modality.DEMONSTRATION,
            query,
            Choice(traj, vending_machine_trajectories)
        )
        return f
    
    def preference(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        r = [qi for qi in query.trajectories]
        items = []
        for traj in r:
            closest = float("inf")
            closest_item = None
            # print("phi: ", traj.phi, "length: ", np.linalg.norm(traj.phi))
            for item in self.vending_machine_items.values():
                dist = np.linalg.norm(traj.phi - item.get_features())
                # print("dist: ", dist, "item: ", item.itemName)
                if dist < closest:
                    closest = dist
                    closest_item = item.itemName
            items.append(closest_item)
            print("-----------------")               

        r.append(Trajectory(states=self.vending_machine_items["Water"].get_features(), actions=None, phi=self.vending_machine_items["Water"].get_features()))
        print("These are your choices: ", items, "Water")#, random_items)

        fav = input("Please provide your preference: ")
        if fav == "0":
            print("You've chosen: ", items[0])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[0], r)
            )
        elif fav == "1":
            print("You've chosen: ", items[1])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[1], r)
            )
        elif fav == "2":
            print("You've chosen: ", "Water")
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[2], r)
            )
        else:
            print("I'm sorry, I didn't understand that.")
    
    # TODO - query for a food, but ask for a correction that should be another food
    def correction(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        pass

    def binary_feedback(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        pass

    


        # get the best trajectory from the task
        # best_traj = task.get_best_trajectory(query.start_state)
        # # get the worst trajectory from the task
        # worst_traj = task.get_worst_trajectory(query.start_state)
        # # get the trajectory samples from the task
        # traj_samples = task.get_trajectory_samples(query.start_state)
        # # create a feedback object
        # f = Feedback(
        #     Modality.DEMONSTRATION,
        #     query,
        #     Choice(best_traj, [best_traj, worst_traj, traj_samples])
        # )
        # return f