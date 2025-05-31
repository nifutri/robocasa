import argparse
import json
import os
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored

import robocasa

import numpy as np
import copy
import os
import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robocasa.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

# from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, get_env_from_dataset
from robomimic.config import config_factory
import pdb
import robosuite

# import robocasa
# from robocasa.utils.env_utils import create_env, run_random_rollouts
import sys

# import os
import pathlib

# ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
# sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    if render is False:
        print(colored("Running episode...", "yellow"))

    for i in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    if verbose or i == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, i
                                ),
                                "yellow",
                            )
                        )
        else:
            reset_to(env, {"states": states[i]})

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions and not args.use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    if not args.use_obs:
        # # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # # for playback since observations are unused. Pass a dummy spec here.
        # dummy_spec = dict(
        #     obs=dict(
        #             low_dim=["robot0_eef_pos"],
        #             rgb=[],
        #         ),
        # )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = False

        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )

        env = robosuite.make(**env_kwargs)
        import pdb

        pdb.set_trace()

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif args.use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )

    f.close()
    if write_video:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()


def get_env_from_dataset(dataset):
    # args = get_playback_args()
    # # some arg checking
    # write_video = args.render is not True
    # if args.video_path is None:
    #     args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
    #     if args.use_actions:
    #         args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
    #     elif args.use_abs_actions:
    #         args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    # assert not (args.render and write_video)  # either on-screen or video but not both

    # # Auto-fill camera rendering info if not specified
    # if args.render_image_names is None:
    #     # We fill in the automatic values
    #     env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
    #     args.render_image_names = "robot0_agentview_center"

    # if args.render:
    #     # on-screen rendering can only support one camera
    #     assert len(args.render_image_names) == 1

    # if args.use_obs:
    #     assert write_video, "playback with observations can only write to video"
    #     assert (
    #         not args.use_actions and not args.use_abs_actions
    #     ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    # if not args.use_obs:

    env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
    # if args.use_abs_actions:
    env_meta["env_kwargs"]["controller_configs"][
        "control_delta"
    ] = False  # absolute action space

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False

    # if args.verbose:
    #     print(
    #         colored(
    #             "Initializing environment for {}...".format(env_kwargs["env_name"]),
    #             "yellow",
    #         )
    #     )

    env = robosuite.make(**env_kwargs)
    return env


def get_playback_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) width of image observations",
    )

    args = parser.parse_args()
    return args


class RobocasaAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name="bc"):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = get_env_metadata_from_dataset(dataset_path)
        print("env_meta = {}".format(env_meta))
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        # setup env arguments
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["use_camera_obs"] = False
        # pdb.set_trace()
        # env = get_env_from_dataset(dataset_path)
        # change directories to robocasa
        # get current directory
        # curr_dir = os.getcwd()
        # os.chdir(os.path.dirname('../../../robocasa/robocasa/scripts/'))

        # env_kwargs={'env_name': 'CloseDrawer', 'robots': 'PandaMobile', 'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}, 'layout_ids': -1, 'style_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 11], 'translucent_robot': False, 'obj_instance_split': 'A', 'has_renderer': False, 'renderer': 'mjviewer', 'has_offscreen_renderer': False, 'use_camera_obs': False}

        # pdb.set_trace()
        env = robosuite.make(**env_kwargs)
        # display robosuite path and version
        # robosuite_path = robosuite.__file__
        # robosuite_version = robosuite.__version__
        # print(f"Robosuite path: {robosuite_path}")

        # env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
        #     render=False,
        #     render_offscreen=False,
        #     use_image_obs=False,
        # )
        # pdb.set_trace()
        assert len(env.robots) in (1, 2)
        abs_env_kwargs = abs_env_meta["env_kwargs"]
        abs_env_kwargs["env_name"] = abs_env_meta["env_name"]
        abs_env_kwargs["has_renderer"] = False
        abs_env_kwargs["renderer"] = "mjviewer"
        abs_env_kwargs["has_offscreen_renderer"] = False
        abs_env_kwargs["use_camera_obs"] = False
        abs_env = robosuite.make(**env_kwargs)

        # abs_env = EnvUtils.create_env_from_metadata(
        #     env_meta=abs_env_meta,
        #     render=False,
        #     render_offscreen=False,
        #     use_image_obs=False,
        # )
        # pdb.set_trace()
        # assert not abs_env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.file["data"])

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        print("states.shape = {}".format(states.shape))
        print("actions.shape = {}".format(actions.shape))
        print("actions = {}".format(actions))
        # pdb.set_trace()
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 12)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_gripper = stacked_actions[..., [-1]]
        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i, idx], policy_step=True)

                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i, idx] = controller.goal_pos
                action_goal_ori[i, idx] = Rotation.from_matrix(
                    controller.goal_ori
                ).as_rotvec()

        stacked_abs_actions = np.concatenate(
            [action_goal_pos, action_goal_ori, action_gripper], axis=-1
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
        robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

        delta_error_info = self.evaluate_rollout_error(
            env,
            states,
            actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )
        abs_error_info = self.evaluate_rollout_error(
            abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )

        info = {"delta_max_error": delta_error_info, "abs_max_error": abs_error_info}
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(
        env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=1
    ):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({"states": states[0]})
        for i in range(len(states)):
            obs = env.reset_to({"states": states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()["states"])
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = (
            Rotation.from_quat(robot0_eef_quat[1:])
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        )
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            "state": max_next_state_diff,
            "pos": max_next_eef_pos_dist,
            "rot": max_next_eef_rot_dist,
        }
        return info


def worker(x):
    path, idx, do_eval = x
    converter = RobocasaAbsoluteActionConverter(path)
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = dict()
    return abs_actions, info


@click.command()
@click.option("-i", "--input", required=True, help="input hdf5 path")
@click.option(
    "-o",
    "--output",
    required=True,
    help="output hdf5 path. Parent directory must exist",
)
@click.option(
    "-e", "--eval_dir", default=None, help="directory to output evaluation metrics"
)
@click.option("-n", "--num_workers", default=None, type=int)
def main(input, output, eval_dir, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()

    # open input file
    # with h5py.File(input, 'r') as f:
    #     data = f['data']
    #     pdb.set_trace()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()
        do_eval = True

    converter = RobocasaAbsoluteActionConverter(input)

    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(input, i, do_eval) for i in range(len(converter))])

    # save output
    print("Copying hdf5")
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, "r+") as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file[f"data/demo_{i}"]
            demo["actions"][:] = abs_actions

    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath("error_stats.pkl").open("wb"))

        print("Generating visualization")
        metrics = ["pos", "rot"]
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt

        plt.switch_backend("PDF")

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10, 4)
        fig.savefig(str(eval_dir.joinpath("error_stats.pdf")))
        fig.savefig(str(eval_dir.joinpath("error_stats.png")))


if __name__ == "__main__":
    # pdb.set_trace()
    main()
