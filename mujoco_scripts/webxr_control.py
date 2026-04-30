# Author: Jimmy Wu
# Date: October 2024

import json
import logging
import math
import os
import threading
import time
from argparse import ArgumentParser
from queue import Empty, Queue

import numpy as np
import zmq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R

TELEOP_HOST = '0.0.0.0'  # bind all interfaces (needed for ngrok tunnel)

# Optional static ngrok domain. Leave unset for a generated ngrok URL.
NGROK_DOMAIN = os.environ.get('INSTANT_POLICY_NGROK_DOMAIN')

# ZMQ topic ports
ZMQ_PUB_PORT = 5555  # Single PUB socket, topic-prefixed messages


class ZmqPublisher:
    def __init__(self, port=ZMQ_PUB_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f'tcp://*:{port}')
        # Allow subscribers time to connect
        time.sleep(0.1)

    def publish(self, topic, data):
        msg = json.dumps({'topic': topic, 'data': data, 'timestamp': time.time()})
        self.socket.send_string(f'{topic} {msg}')


class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError

class WebServer:
    def __init__(self, queue):
        self.app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
        self.socketio = SocketIO(self.app)
        self.queue = queue

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.after_request
        def add_no_cache_headers(response):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        @self.socketio.on('message')
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit('echo', data.get('timestamp', 0))

            # Tag with server-side receive time for reliable staleness check
            data['_recv_time'] = time.time()

            # Debug: log message type
            if 'state_update' in data:
                print(f'[WebServer] state_update: {data["state_update"]}')
                if 'debug_msg' in data:
                    print(f'[WebServer] DEBUG from client: {data["debug_msg"]}')
            elif 'teleop_mode' in data:
                print(f'[WebServer] teleop msg: mode={data["teleop_mode"]}, '
                      f'pos=({data["position"]["x"]:.3f}, {data["position"]["y"]:.3f}, {data["position"]["z"]:.3f})')

            # Add data to queue for processing
            self.queue.put(data)

        # Reduce verbose Flask log output
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self):
        port = 5001
        # Start ngrok tunnel automatically
        try:
            from pyngrok import ngrok
            tunnel_kwargs = {'bind_tls': True}
            if NGROK_DOMAIN:
                tunnel_kwargs['domain'] = NGROK_DOMAIN
            tunnel = ngrok.connect(port, **tunnel_kwargs)
            print(f'ngrok tunnel: {tunnel.public_url}')
        except Exception as e:
            print(f'ngrok failed ({e}), falling back to local only')

        print(f'Starting server at {TELEOP_HOST}:{port}')
        self.socketio.run(self.app, host=TELEOP_HOST, port=port)

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro


# Convert coordinate system from WebXR to robot
def convert_webxr_pose(pos, quat):
    # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

    # Apply offset so that rotations are around device center instead of device camera
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot

TWO_PI = 2 * math.pi


class TeleopController:
    def __init__(self):
        self.teleop_enabled = False
        self.enabled_count = 0

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.teleop_mode = None
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

        # Camera frame basis for camera-relative control (set externally)
        self.cam_basis = None

    def process_message(self, data):
        if not self.targets_initialized:
            print('[TeleopController] targets not initialized, dropping message')
            return

        # Update enabled count; reset when teleop is deactivated
        if 'teleop_mode' in data:
            self.enabled_count += 1
        else:
            if self.teleop_enabled:
                self.teleop_enabled = False
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            self.enabled_count = 0

        # Skip first 2 steps: WebXR pose updates have higher latency than touch events
        if self.enabled_count > 2:
            if not self.teleop_enabled:
                print(f'[TeleopController] teleop ENABLED (enabled_count={self.enabled_count})')
            self.teleop_enabled = True

        # Teleop is enabled
        if self.teleop_enabled and 'teleop_mode' in data:
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])
            self.teleop_mode = data['teleop_mode']
            self.data_time = time.time()

            # Base movement
            if data['teleop_mode'] == 'base':
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
                self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data['teleop_mode'] == 'arm':
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                pos_diff = pos - self.arm_xr_ref_pos

                if self.cam_basis is not None:
                    # Camera-relative control: phone delta is interpreted in the
                    # viewer camera frame (forward/left/up) and mapped to world.
                    # cam_basis is a proper rotation (det=+1), so this works
                    # correctly for both position and rotation vectors.
                    self.arm_target_pos = self.arm_ref_pos + self.cam_basis @ pos_diff

                    delta_rotvec = (rot * self.arm_xr_ref_rot_inv).as_rotvec()
                    world_rotvec = self.cam_basis @ delta_rotvec
                    self.arm_target_rot = R.from_rotvec(world_rotvec) * self.arm_ref_rot
                else:
                    # Original base-relative control (for mobile base robots)
                    z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                    z_rot_inv = z_rot.inv()
                    ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

                    self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)
                    self.arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

                # Gripper position
                self.gripper_target_pos = np.clip(self.gripper_ref_pos + data['gripper_delta'], 0.0, 1.0)

        # Teleop is disabled
        elif not self.teleop_enabled:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs):
        # Update robot state
        self.base_pose = obs['base_pose']

        # Update camera basis if provided (for camera-relative control)
        if 'cam_basis' in obs:
            self.cam_basis = obs['cam_basis']

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos']
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos']
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if not self.teleop_enabled:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            'base_pose': self.base_target_pose.copy(),
            'teleop_mode': self.teleop_mode,
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
            'data_time': self.data_time,
        }

        return action


# Teleop using WebXR phone web app
class TeleopPolicy(Policy):
    def __init__(self):
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self.teleop_state = None  # States: episode_ready -> episode_started -> episode_ended -> reset_env
        self.episode_ended = False

        # Web server for serving the WebXR phone web app
        server = WebServer(self.web_server_queue)
        threading.Thread(target=server.run, daemon=True).start()

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def reset(self):
        while True:
            try:
                self.web_server_queue.get_nowait()
            except Empty:
                break

        self.teleop_controller = TeleopController()
        self.episode_ended = False
        self.teleop_state = None

    def step(self, obs):
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == 'reset_env':
            return 'reset_env'

        return self._step(obs)

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def has_live_tracking(self):
        return (
            self.teleop_controller is not None
            and self.teleop_controller.teleop_enabled
        )

    def listener_loop(self):
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()

                # Update state
                if 'state_update' in data:
                    self.teleop_state = data['state_update']

                # Process message if not stale (use server-side receive time to avoid clock drift)
                elif time.time() - data.get('_recv_time', time.time()) < 1.0:
                    self._process_message(data)

            time.sleep(0.001)

    def _process_message(self, data):
        self.teleop_controller.process_message(data)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    control_period = float(os.environ.get('POLICY_CONTROL_PERIOD', '0.05'))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pub = ZmqPublisher(port=ZMQ_PUB_PORT)
    logger.info("Teleoperation policy test started (ZMQ publisher on port %d)", ZMQ_PUB_PORT)

    obs = {
        'base_pose': np.zeros(3),
        'arm_pos': np.zeros(3),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'gripper_pos': np.zeros(1),
        'base_image': np.zeros((640, 360, 3)),
        'wrist_image': np.zeros((640, 480, 3)),
    }
    policy = TeleopPolicy()
    policy.reset()
    pub.publish('/teleop_start', {'data': True})

    prev_twist_cmd = None
    prev_target_pos = None
    prev_target_rot = None
    prev_data_time = None

    while True:
        action = policy.step(obs)
        if action == 'end_episode':
            logger.info("Episode ended by user")
            continue
        elif action == 'reset_env':
            logger.info("Environment reset requested by user")
            pub.publish('/teleop_reset', {'data': True})
            # Stop motion: zero twist and hold last position
            pub.publish('/pin_twist_cmd', {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            })
            if prev_target_pos is not None:
                hold_quat = prev_target_rot.as_quat()
                pub.publish('/pin_pos_cmd', {
                    'stamp': time.time(),
                    'frame_id': 'base_link',
                    'position': {'x': float(prev_target_pos[0]), 'y': float(prev_target_pos[1]), 'z': float(prev_target_pos[2])},
                    'orientation': {'x': float(hold_quat[0]), 'y': float(hold_quat[1]), 'z': float(hold_quat[2]), 'w': float(hold_quat[3])},
                })
            policy.reset()
            pub.publish('/teleop_start', {'data': True})
            prev_target_pos, prev_target_rot = None, None
            prev_twist_cmd = None
            continue
        else:
            pub.publish('/pin_ctrl_enable', {'data': True})

        if action is None:
            logger.info("Action was none")
            time.sleep(control_period)
            prev_target_pos, prev_target_rot = None, None
            continue

        current_pos = action['arm_pos'].copy()
        current_quat = action['arm_quat'].copy()
        current_rot = R.from_quat(current_quat)

        # Rotate 180 degrees around z-axis
        #z180 = R.from_rotvec([0.0, 0.0, math.pi])
        #current_pos = z180.apply(current_pos)
        #current_rot = z180 * current_rot
        current_quat = current_rot.as_quat()

        pub.publish('/pin_pos_cmd', {
            'stamp': time.time(),
            'frame_id': 'base_link',
            'position': {'x': float(current_pos[0]), 'y': float(current_pos[1]), 'z': float(current_pos[2])},
            'orientation': {'x': float(current_quat[0]), 'y': float(current_quat[1]), 'z': float(current_quat[2]), 'w': float(current_quat[3])},
        })
        pub.publish('/pin_gripper_cmd', {'data': float(action['gripper_pos'][0] * 0.1)})  # Scale to [0, 0.1] m

        if prev_target_pos is not None:
            delta_linear = current_pos - prev_target_pos
            delta_angular = (current_rot * prev_target_rot.inv()).as_euler('xyz')

            max_linear_vel = 1.0  # m/s
            max_angular_vel = 4.0  # rad/s

            period = np.max([control_period, action['data_time'] - prev_data_time])
            delta_linear = np.clip(delta_linear / period, -max_linear_vel, max_linear_vel)
            delta_angular = np.clip(delta_angular / period, -max_angular_vel, max_angular_vel)

            twist_cmd = {
                'linear': {'x': float(delta_linear[0]), 'y': float(delta_linear[1]), 'z': float(delta_linear[2])},
                'angular': {'x': float(delta_angular[0]), 'y': float(delta_angular[1]), 'z': float(delta_angular[2])},
            }

            all_zero = all(abs(v) < 1e-8 for v in delta_linear) and all(abs(v) < 1e-8 for v in delta_angular)

            if all_zero and prev_twist_cmd is not None:
                pub.publish('/pin_twist_cmd', prev_twist_cmd)
            else:
                pub.publish('/pin_twist_cmd', twist_cmd)
            logger.info("Published twist command: %s", twist_cmd)
            prev_twist_cmd = twist_cmd

        prev_target_pos = current_pos.copy()
        prev_target_rot = R.from_quat(current_quat.copy())
        prev_data_time = action['data_time']

        time.sleep(control_period)  # Note: Not precise
