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
        print("vending machine: ", vending_machine.food_items)
        self.vending_machine_items = vending_machine.food_items
    
    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        if q.query_type is Modality.DEMONSTRATION:
            f = self.demonstration(q, task)
            print("the demonstration: ", f.choice.selection.phi)
            return f
        elif q.query_type is Modality.PREFERENCE:
            f = self.preference(q, task)
            print("the preference: ", f.choice.selection.phi)
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

        print("traj: ", traj.phi)

        f = Feedback(
            Modality.DEMONSTRATION,
            query,
            Choice(traj, [traj] + query.trajectories)
        )
        return f
    
    # TODO - make the preference actual items
    def preference(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        r = [qi for qi in query.trajectories]
        items = []
        for traj in r:
            # print("States!!: ", traj.states)
            # print("Phi: ", traj.phi)
            closest = float("inf")
            closest_item = None
            for item in self.vending_machine_items.values():
                dist = np.linalg.norm(traj.phi - item.get_features())
                if dist < closest:
                    closest = dist
                    closest_item = item.itemName
            items.append(closest_item)               

        # give me three random numbers between 0 and 20
        j = np.random.choice(20, 3, replace=False)
        # get three random item names from the vending machine
        random_items = [list(self.vending_machine_items.keys())[i] for i in j]
                # print("these are the options: ", r)
        r = [Trajectory(states=self.vending_machine_items[items[0]].get_features(), actions=None, phi=self.vending_machine_items[items[0]].get_features()),
             Trajectory(states=self.vending_machine_items[items[1]].get_features(), actions=None, phi=self.vending_machine_items[items[1]].get_features()),
            ]
        
        for it in random_items:
            r.append(Trajectory(states=self.vending_machine_items[it].get_features(), actions=None, phi=self.vending_machine_items[it].get_features()))
        
        print("These are your choices: ", items, random_items)

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
            print("You've chosen: ", random_items[0])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[2], r)
            )
        elif fav == "3":
            print("You've chosen: ", random_items[1])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[3], r)
            )
        elif fav == "4":
            print("You've chosen: ", random_items[2])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[4], r)
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