"""
Spawn pedestrians with specific crossing behaviors to test AEB.
"""

import time
import random
import carla


class ScenarioManager:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.spawned_actors = []

        self.bp_lib = self.world.get_blueprint_library()
        self.walker_bps = self.bp_lib.filter("walker.pedestrian.*")
        self.vehicle_bps = self.bp_lib.filter("vehicle.*")
        self.controller_bp = self.bp_lib.find("controller.ai.walker")

    def _get_ego_transform(self):
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle is not available")
        return self.ego_vehicle.get_transform()

    def _spawn_walker_with_controller(self, location, yaw=0.0, speed=1.4):
        walker_bp = random.choice(self.walker_bps)
        walker_transform = carla.Transform(location, carla.Rotation(yaw=yaw))

        walker = self.world.try_spawn_actor(walker_bp, walker_transform)
        if walker is None:
            raise RuntimeError(f"Failed to spawn walker at {location}")

        controller = self.world.try_spawn_actor(
            self.controller_bp,
            carla.Transform(),
            attach_to=walker
        )
        if controller is None:
            walker.destroy()
            raise RuntimeError("Failed to spawn walker controller")

        self.spawned_actors.extend([walker, controller])

        # one tick helps CARLA register actors properly in sync mode
        self.world.tick()

        controller.start()
        controller.set_max_speed(float(speed))
        return walker, controller

    def spawn_crossing_pedestrian(self, crossing_speed=1.4, start_offset=10.0):
        """
        Scenario 1: Pedestrian crosses road in front of ego vehicle.
        """
        ego_tf = self._get_ego_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        # Spawn ahead of ego and slightly to one side
        spawn_loc = ego_tf.location + carla.Location(
            x=fwd.x * start_offset + right.x * 4.0,
            y=fwd.y * start_offset + right.y * 4.0,
            z=0.5
        )

        target_loc = ego_tf.location + carla.Location(
            x=fwd.x * (start_offset + 2.0) - right.x * 4.0,
            y=fwd.y * (start_offset + 2.0) - right.y * 4.0,
            z=0.5
        )

        walker, controller = self._spawn_walker_with_controller(
            spawn_loc, yaw=ego_tf.rotation.yaw + 90.0, speed=crossing_speed
        )

        controller.go_to_location(target_loc)
        return {
            "scenario": "crossing_pedestrian",
            "walker": walker,
            "controller": controller,
            "start": spawn_loc,
            "target": target_loc,
            "speed_mps": crossing_speed,
        }

    def spawn_child_behind_car(self, start_offset=12.0, parked_offset=10.0):
        """
        Scenario 2: Child appears suddenly from behind parked car.
        """
        ego_tf = self._get_ego_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        # parked car on roadside
        parked_loc = ego_tf.location + carla.Location(
            x=fwd.x * parked_offset + right.x * 4.0,
            y=fwd.y * parked_offset + right.y * 4.0,
            z=0.5
        )
        parked_tf = carla.Transform(parked_loc, ego_tf.rotation)

        parked_bp = random.choice(self.vehicle_bps)
        parked_car = self.world.try_spawn_actor(parked_bp, parked_tf)
        if parked_car is None:
            raise RuntimeError("Failed to spawn parked car")

        self.spawned_actors.append(parked_car)

        # child/walker behind parked car
        child_loc = ego_tf.location + carla.Location(
            x=fwd.x * start_offset + right.x * 5.0,
            y=fwd.y * start_offset + right.y * 5.0,
            z=0.5
        )
        target_loc = ego_tf.location + carla.Location(
            x=fwd.x * (start_offset + 1.0) - right.x * 3.0,
            y=fwd.y * (start_offset + 1.0) - right.y * 3.0,
            z=0.5
        )

        walker, controller = self._spawn_walker_with_controller(
            child_loc, yaw=ego_tf.rotation.yaw + 90.0, speed=3.0
        )

        time.sleep(1.0)
        controller.go_to_location(target_loc)

        return {
            "scenario": "child_behind_car",
            "parked_car": parked_car,
            "walker": walker,
            "controller": controller,
            "start": child_loc,
            "target": target_loc,
            "speed_mps": 3.0,
        }

    def spawn_standing_pedestrian(self, distance=30.0):
        """
        Scenario 3: Pedestrian standing near road edge.
        """
        ego_tf = self._get_ego_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        spawn_loc = ego_tf.location + carla.Location(
            x=fwd.x * distance + right.x * 4.5,
            y=fwd.y * distance + right.y * 4.5,
            z=0.5
        )

        walker_bp = random.choice(self.walker_bps)
        walker_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_tf.rotation.yaw))
        walker = self.world.try_spawn_actor(walker_bp, walker_tf)

        if walker is None:
            raise RuntimeError("Failed to spawn standing pedestrian")

        self.spawned_actors.append(walker)

        return {
            "scenario": "standing_pedestrian",
            "walker": walker,
            "location": spawn_loc,
        }

    def spawn_running_pedestrian(self, start_offset=10.0):
        """
        Scenario 4: Pedestrian running across road at ~8 km/h.
        8 km/h ≈ 2.22 m/s
        """
        return self.spawn_crossing_pedestrian(crossing_speed=2.22, start_offset=start_offset)

    def setup_school_zone(self, crossing_speed=1.2, start_offset=8.0):
        """
        Scenario 5: School zone style scenario.
        Sign asset placement is optional because many CARLA maps do not expose easy sign spawning.
        This function at least creates a slower crossing scenario closer to ego vehicle.
        """
        scenario = self.spawn_crossing_pedestrian(
            crossing_speed=crossing_speed,
            start_offset=start_offset
        )
        scenario["scenario"] = "school_zone_crossing"
        scenario["speed_zone_kmh"] = 30
        return scenario

    def cleanup(self):
        """Destroy all spawned scenario actors."""
        for actor in reversed(self.spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned_actors = []