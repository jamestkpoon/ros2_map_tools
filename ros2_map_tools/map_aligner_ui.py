#!/usr/bin/env python3

import pathlib
from copy import deepcopy
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from os.path import exists
from typing import Dict, Iterable

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

WINDOW_NAME = "map_aligner"
POLL_DUR = 0.05
WAITKEY_MS = 20

SUBMAP_BREAK_KEY = "break"
SUBMAP_DISCARD_KEY = "discard"
TERM_KEYS = ("exit", "quit")


def imshow_loop(conn: Connection):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        if conn.poll(POLL_DUR):
            obj_in = conn.recv()
            if isinstance(obj_in, np.ndarray):
                cv2.imshow(WINDOW_NAME, obj_in)
            else:
                break
        else:
            cv2.waitKey(WAITKEY_MS)

    cv2.destroyWindow(WINDOW_NAME)


def _rot_affine(ccw_deg: float, shape: Iterable[int]):
    center = (shape[1] / 2.0, shape[0] / 2.0)
    rot = cv2.getRotationMatrix2D(center, ccw_deg, 1.0)

    box = np.int0(cv2.boxPoints((center, (shape[1], shape[0]), -ccw_deg)))
    box -= np.min(box, axis=0)
    new_shape = np.max(box, axis=0)

    rot[0, 2] += (new_shape[0] - shape[1]) / 2
    rot[1, 2] += (new_shape[1] - shape[0]) / 2

    return (rot, new_shape, box)


def _transform_matrix_from_2d_pose(pose: Iterable[float]):
    tm = np.eye(4)
    tm[0, 3] = pose[0]
    tm[1, 3] = pose[1]
    tm[:3, :3] = Rotation.from_euler("z", pose[2]).as_matrix()
    return tm


def _transform_px_to_pos(pp, px: Iterable[float], res: float):
    cosyaw = np.cos(pp[1][2])
    sinyaw = np.sin(pp[1][2])
    r = np.asarray([[cosyaw, -sinyaw], [sinyaw, cosyaw]])

    dx = (px[0] - pp[0][0]) * res
    dy = -(px[1] - pp[0][1]) * res
    pos = np.matmul(r, (dx, dy))

    return pp[1][:2] + pos


class Map:
    def __init__(self, yaml_fp: str, resolution_override: float = None):
        self.metadata_: dict = yaml.safe_load(open(yaml_fp, "r", encoding="utf-8"))
        self.md_fp_ = pathlib.Path(yaml_fp)

        self.maps_: Dict[str, np.ndarray] = {
            "map": cv2.imread(
                str(self.md_fp_.parent.joinpath(self.metadata_["image"])),
                cv2.IMREAD_GRAYSCALE,
            )
        }

        if (
            isinstance(resolution_override, float)
            and resolution_override != self.metadata_["resolution"]
        ):
            scaling_factor = self.metadata_["resolution"] / resolution_override
            self.maps_["map"] = cv2.resize(
                self.maps_["map"],
                dsize=(0, 0),
                fx=scaling_factor,
                fy=scaling_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            self.res_ = resolution_override
        else:
            self.res_ = float(self.metadata_["resolution"])

        mask_map = (
            self.maps_["map"] if self.metadata_["negate"] else 255 - self.maps_["map"]
        )
        self.maps_["free"] = mask_map < round(255 * self.metadata_["free_thresh"])
        self.maps_["occupied"] = mask_map > round(
            255 * self.metadata_["occupied_thresh"]
        )
        self.maps_["unknown"] = np.logical_not(
            np.logical_and(self.maps_["free"], self.maps_["occupied"])
        )

    @property
    def metadata(self):
        return deepcopy(self.metadata_)

    @property
    def res(self):
        return self.res_

    @property
    def shape(self):
        return self.maps_["map"].shape

    @property
    def free_color(self):
        return 0 if self.metadata["negate"] else 255

    @property
    def occupied_color(self):
        return 255 - self.free_color

    @property
    def unknown_color(self):
        return round(
            255
            * (self.metadata_["free_thresh"] + self.metadata_["occupied_thresh"])
            / 2
        )

    @property
    def occupied_threshold_color(self):
        v = min(255, int(np.ceil(self.metadata_["occupied_thresh"] * 255)) + 1)
        return v if self.metadata_["negate"] else (255 - v)

    @property
    def unrotated_bl_px_pose(self):
        return self.transform_bl_px_pose(
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float), 0.0
        )

    def get_map(self, key: str):
        return self.maps_[key]

    def transform_bl_px_pose(self, m: np.ndarray, ccw_deg: float):
        origin = np.matmul(
            m,
            (
                -self.metadata_["origin"][0] / self.res_,
                self.shape[0] - self.metadata_["origin"][1] / self.res_,
                1,
            ),
        )
        bl = np.matmul(m, (0, self.shape[0], 1))[:2]

        bl_pose = np.asarray(self.metadata_["origin"]) + (
            bl[0] * self.res_,
            -(bl[1] - self.shape[0]) * self.res_,
            np.deg2rad(ccw_deg),
        )

        return (bl, bl_pose)

    def write(self, origin: Iterable[float]):
        metadata = self.metadata
        metadata["origin"] = list(map(float, origin))

        if isinstance(tf := metadata.get("transform"), dict):
            self._apply_new_origin_to_transform(tf, metadata["origin"])

        fp = str(self.md_fp_)
        fp = fp[: fp.rfind(".")] + "_aligned.yaml"
        yaml.dump(metadata, open(fp, "w", encoding="utf-8"))

        return fp

    def _apply_new_origin_to_transform(self, tf: dict, origin: Iterable[float]):
        tm = np.eye(4)
        tm[0, 3] = tf["translation"]["x"]
        tm[1, 3] = tf["translation"]["y"]
        tm[2, 3] = tf["translation"]["z"]
        tm[:3, :3] = Rotation.from_quat(
            (
                tf["rotation"]["x"],
                tf["rotation"]["y"],
                tf["rotation"]["z"],
                tf["rotation"]["w"],
            )
        ).as_matrix()

        origin_tm = _transform_matrix_from_2d_pose(origin)
        old_origin_inv = _transform_matrix_from_2d_pose(self.metadata_["origin"])
        old_origin_inv[:3, :3] = old_origin_inv[:3, :3].T
        old_origin_inv[:3, 3] = -np.matmul(
            old_origin_inv[:3, :3], old_origin_inv[:3, 3]
        )
        tm = np.matmul(np.matmul(origin_tm, old_origin_inv), tm)
        quat = Rotation.from_matrix(tm[:3, :3]).as_quat()

        tf["translation"]["x"] = float(tm[0, 3])
        tf["translation"]["y"] = float(tm[1, 3])
        tf["translation"]["z"] = float(tm[2, 3])
        tf["rotation"]["x"] = float(quat[0])
        tf["rotation"]["y"] = float(quat[1])
        tf["rotation"]["z"] = float(quat[2])
        tf["rotation"]["w"] = float(quat[3])


def _get_valid_filepath(prompt: str):
    while True:
        fp = input(prompt)
        if exists(fp):
            return fp
        else:
            print("Path not found, please try again")


def _parse_adjustment_cmd(cmd: str):
    try:
        s = cmd.split(" ")
        return (s[0].lower(), float(s[1]))
    except:
        return (None, None)


if __name__ == "__main__":
    # start imshow process
    conn_a, conn_b = Pipe()
    imshow_process = Process(target=imshow_loop, args=(conn_b,), daemon=True)
    imshow_process.start()

    # target submap and combined metadata
    user_input = _get_valid_filepath("Target .yaml: ")
    target = Map(user_input)
    target_blpp = target.unrotated_bl_px_pose
    metadata = target.metadata
    if "map" in metadata:
        del metadata["map"]
        del metadata["transform"]

    # start from target map
    combined_map = np.empty(target.shape, dtype=np.uint8)
    combined_map.fill(target.unknown_color)
    combined_map[target.get_map("free")] = target.free_color
    combined_map[target.get_map("occupied")] = target.occupied_color
    conn_a.send(combined_map)

    # align submaps
    print("Some example adjustment commands: 'x 430', 'y -31', 'r 0.2'")
    while True:
        user_input = input("Source .yaml or " + str(TERM_KEYS) + ": ")
        if user_input.lower() in TERM_KEYS:
            break
        if not exists(user_input):
            continue

        source = Map(user_input, target.res)
        source_blpp = source.unrotated_bl_px_pose
        source_free = 255 * source.get_map("free")
        source_occ = 255 * source.get_map("occupied")

        while True:
            # get rotated submap properties, and rotated masks
            yaw = np.rad2deg(source_blpp[1][2] - target_blpp[1][2])
            rotm, shape_xy, border = _rot_affine(yaw, source.shape)
            source_free_rotated = cv2.warpAffine(
                source_free,
                rotm,
                shape_xy,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            source_occ_rotated = cv2.warpAffine(
                source_occ,
                rotm,
                shape_xy,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # create new combined map
            source_center = np.asarray(
                (
                    source_blpp[0][0] + source.shape[1] / 2.0,
                    source_blpp[0][1] - source.shape[0] / 2.0,
                )
            )
            source_ul = source_center - shape_xy / 2
            cm_br = cv2.boundingRect(
                np.round(
                    np.asarray(
                        (
                            source_ul,
                            source_center + shape_xy / 2,
                            (0, 0),
                            (combined_map.shape[1], combined_map.shape[0]),
                        )
                    )
                ).astype(int)
            )
            cm_with_smap = np.empty((cm_br[3], cm_br[2]), np.uint8)
            cm_with_smap.fill(target.unknown_color)

            # slices
            cm_old_slice = np.index_exp[
                -cm_br[1] : -cm_br[1] + combined_map.shape[0],
                -cm_br[0] : -cm_br[0] + combined_map.shape[1],
            ]
            source_ul = np.round(np.max(((0, 0), source_ul), axis=0)).astype(int)
            source_slice = np.index_exp[
                source_ul[1] : source_ul[1] + shape_xy[1],
                source_ul[0] : source_ul[0] + shape_xy[0],
            ]

            # draw free regions and prior occupied
            cm_with_smap[cm_old_slice][
                combined_map == target.free_color
            ] = target.free_color
            cm_with_smap_ss = cm_with_smap[source_slice]
            cm_with_smap_ss[source_free_rotated > 0] = target.free_color
            cm_with_smap[cm_old_slice][
                combined_map == target.occupied_color
            ] = target.occupied_threshold_color

            # draw tentatively occupied
            prior_occ_roi = cm_with_smap_ss == target.occupied_threshold_color
            source_occ_roi = source_occ_rotated > 0
            cm_with_smap_ss[
                np.logical_and(prior_occ_roi, source_occ_roi)
            ] = target.occupied_color
            cm_with_smap_ss[
                np.logical_and(~prior_occ_roi, source_occ_roi)
            ] = target.occupied_threshold_color

            # show with submap border
            cm_with_smap_border = cm_with_smap.copy()
            cv2.drawContours(
                cm_with_smap_border,
                [border + source_ul],
                contourIdx=0,
                color=target.occupied_color,
                thickness=1,
            )
            conn_a.send(cm_with_smap_border)

            # handle user input
            user_input = input(
                "  Enter adjust command or "
                + str((SUBMAP_BREAK_KEY, SUBMAP_DISCARD_KEY))
                + ": "
            ).lower()
            if user_input == SUBMAP_BREAK_KEY:
                combined_map = cm_with_smap

                target_blpp[0][:] -= cm_br[:2]
                source_blpp_tf_px = (
                    source_blpp[0]
                    + source.transform_bl_px_pose(rotm, yaw)[0]
                    - (0, source.shape[0])
                )
                source_blpp[1][:2] = _transform_px_to_pos(
                    target_blpp, source_blpp_tf_px, target.res
                )
                source_fp = source.write(source_blpp[1])
                print("Saved " + source_fp)

                break
            if user_input == SUBMAP_DISCARD_KEY:
                print("Submap discarded")
                break

            cmd = _parse_adjustment_cmd(user_input)
            if cmd[0] == "x":
                source_blpp[0][0] += cmd[1]
            elif cmd[0] == "y":
                source_blpp[0][1] += cmd[1]
            elif cmd[0] == "r":
                source_blpp[1][2] += cmd[1]

    # save combined map
    cmn = input("Enter combined map name, or leave empty to discard: ")
    if len(cmn) > 0:
        origin_pose = list(
            map(
                float,
                _transform_px_to_pos(
                    target_blpp, (0, combined_map.shape[0]), target.res
                ),
            )
        )
        origin_pose.append(float(target_blpp[1][2]))

        img_ext = metadata["image"][metadata["image"].rfind(".") :]
        metadata["image"] = cmn + img_ext
        metadata["origin"] = origin_pose
        yaml.dump(metadata, open(cmn + ".yaml", "w", encoding="utf-8"))

        cv2.imwrite(metadata["image"], combined_map)

        print("Saved combined map")

    # stop imshow process
    conn_a.send(None)
    imshow_process.join()
    imshow_process.close()
