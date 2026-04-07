# src/carla_sim/carla_bridge.py

import queue
import random
import numpy as np

try:
    import carla
except ImportError:
    carla = None


class CARLABridge:
    def __init__(self, host="localhost", port=2000, fps=20):
        if carla is None:
            raise ImportError("carla Python package is not installed")

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.image_queue = queue.Queue()
        self.actors = []

        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        self.world.apply_settings(settings)

        self.ego_vehicle = None
        self.camera = None

    def spawn_ego_vehicle(self, vehicle_type="vehicle.tesla.model3", spawn_index=0, random_spawn=False):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find(vehicle_type)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map")

        if random_spawn:
            spawn_point = random.choice(spawn_points)
        else:
            spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
            spawn_point = spawn_points[spawn_index]

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actors.append(self.ego_vehicle)
        return self.ego_vehicle

    def attach_camera(self, width=640, height=480, fov=90):
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle must be spawned before attaching camera")

        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))

        transform = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-5.0)
        )

        self.camera = self.world.spawn_actor(
            cam_bp, transform, attach_to=self.ego_vehicle
        )
        self.camera.listen(self._camera_callback)
        self.actors.append(self.camera)
        return self.camera

    def _camera_callback(self, image):
        # Keep only fresh frames to avoid lag buildup
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        self.image_queue.put(image)

    def tick(self):
        self.world.tick()

    def get_frame(self, timeout=5.0):
        image = self.image_queue.get(timeout=timeout)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, :3].copy()

    def apply_control(self, throttle=0.3, brake=0.0, steer=0.0):
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle not spawned")

        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer)
        )
        self.ego_vehicle.apply_control(control)

    def apply_brake(self, brake_intensity=1.0):
        self.apply_control(throttle=0.0, brake=brake_intensity, steer=0.0)

    def apply_throttle(self, throttle=0.3):
        self.apply_control(throttle=throttle, brake=0.0, steer=0.0)

    def get_speed_kmh(self):
        if self.ego_vehicle is None:
            return 0.0

        vel = self.ego_vehicle.get_velocity()
        speed_ms = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
        return speed_ms * 3.6

    def get_vehicle_speed(self):
        return self.get_speed_kmh()

    def set_weather(self, weather_preset):
        presets = {
            "Clear": carla.WeatherParameters.ClearNoon,
            "Rain": carla.WeatherParameters.HardRainNoon,
            "Fog": carla.WeatherParameters.FoggyNoon,
            "Night": carla.WeatherParameters.ClearNight,
        }

        if weather_preset not in presets:
            raise ValueError(f"Unknown weather preset: {weather_preset}")

        self.world.set_weather(presets[weather_preset])

    def cleanup(self):
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except Exception:
                pass

        self.actors = []
        self.ego_vehicle = None
        self.camera = None

        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)