
"""
evalate tracking
"""

import numpy as np
import evo.tools.file_interface as FI
import evo.core.trajectory as TJ
from evo.core import lie_algebra as lie
from matplotlib import pyplot as plt


class EvaluateTracking():
    # self.est_stamps # [kf, f]
    # self.est_poses # [relatvie poses]
    # self.gf
    # self.gt_stamps
    # self.gt_abs_poses # [abs poses]

    def __init__(self, gt_file, frame_file, scale):
        self.process_frame(frame_file, scale)  # [kf_stamps, f_stamps]
        self.process_gt(gt_file)
        self.align()
        self.plot_error_ratio()

    def process_frame(self, frame_file, scale):
        raw_mat = FI.csv_read_matrix(frame_file, delim=" ", comment_str="#")
        mat = np.array(raw_mat).astype(float)
        self.est_stamps = mat[:, 0:2]  # n x 1
        xyz = mat[:, 2:5] * scale  # n x 3
        quat = mat[:, 5:9]  # n x 4
        self.gf_ratio = mat[:, 9:]  # n x 1
        # shift 1 column -> w in front column
        quat = np.roll(quat, 1, axis=1)

        self.est_poses = TJ.xyz_quat_wxyz_to_se3_poses(xyz, quat)

    def process_gt(self, gt_file):
        raw_mat = FI.csv_read_matrix(gt_file, delim=" ", comment_str="#")
        mat = np.array(raw_mat).astype(float)
        self.gt_stamps = mat[:, 0]
        xyz = mat[:, 1:4]  # n x 3
        quat = mat[:, 4:]  # n x 4
        # shift 1 column -> w in front column
        quat = np.roll(quat, 1, axis=1)
        self.gt_abs_poses = TJ.xyz_quat_wxyz_to_se3_poses(xyz, quat)

    def align(self):
        max_diff = 0.01  # sec
        self.error = []
        self.ratio = []
        for est_index, stamp in enumerate(self.est_stamps):
            diffs_kf = np.abs(self.gt_stamps - stamp[0])
            index_kf = np.argmin(diffs_kf)
            diffs_f = np.abs(self.gt_stamps - stamp[1])
            index_f = np.argmin(diffs_f)
            if diffs_kf[index_kf] <= max_diff and diffs_f[index_f] <= max_diff:
                gt_pose = lie.relative_se3(
                    self.gt_abs_poses[index_kf], self.gt_abs_poses[index_f])
                E = lie.relative_se3(gt_pose, self.est_poses[est_index])
                if self.gf_ratio[est_index] <= 0.0:
                    continue
                self.error.append(np.linalg.norm(E[:3, 3]))
                self.ratio.append(self.gf_ratio[est_index])

    def plot_error_ratio(self):
        # plot the data
        plt.title('GF ratio V.S. RPE')
        plt.xlabel('RPE (m)')
        plt.ylabel('GF ratio')
        error_array = np.array(self.error)
        ratio_array = np.array(self.ratio)
        plt.plot(error_array, ratio_array, 'b.')
        # plt.plot(timestamps, error, 'b.')
        plt.savefig("./GF_ratio_error.png")
        plt.show()
        plt.close()


class EvaluateTracking2():
    # self.est_stamps # [kf, f]
    # self.est_poses # [relatvie poses]
    # self.gf
    # self.gt_stamps
    # self.gt_abs_poses # [abs poses]

    def __init__(self, gt_file, frame_file1, scale1, frame_file2, scale2):
        stamps1, error1 = self.align(gt_file, frame_file1, scale1)
        stamps2, error2 = self.align(gt_file, frame_file2, scale2)
        self.plot_error(stamps1, error1, stamps2, error2)

    def align(self, gt_file, frame_file, scale):
        est_stamps, est_poses = self.process_frame(frame_file, scale)
        gt_stamps, gt_abs_poses = self.process_gt(gt_file)

        max_diff = 0.01  # sec
        f_errors = []
        f_stamps = []
        for est_index, stamp in enumerate(est_stamps):
            diffs_kf = np.abs(gt_stamps - stamp[0])
            index_kf = np.argmin(diffs_kf)
            diffs_f = np.abs(gt_stamps - stamp[1])
            index_f = np.argmin(diffs_f)
            if diffs_kf[index_kf] <= max_diff and diffs_f[index_f] <= max_diff:
                gt_pose = lie.relative_se3(
                    gt_abs_poses[index_kf], gt_abs_poses[index_f])
                E = lie.relative_se3(gt_pose, est_poses[est_index])
                f_errors.append(np.linalg.norm(E[:3, 3]))
                f_stamps.append(stamp[1])
        return f_stamps, f_errors




    def process_frame(self, frame_file, scale):
        raw_mat = FI.csv_read_matrix(frame_file, delim=" ", comment_str="#")
        mat = np.array(raw_mat).astype(float)
        est_stamps = mat[:, 0:2]  # n x 1
        xyz = mat[:, 2:5] * scale  # n x 3
        quat = mat[:, 5:9]  # n x 4
        gf_ratio = mat[:, 9:]  # n x 1
        # shift 1 column -> w in front column
        quat = np.roll(quat, 1, axis=1)

        est_poses = TJ.xyz_quat_wxyz_to_se3_poses(xyz, quat)
        return est_stamps, est_poses

    def process_gt(self, gt_file):
        raw_mat = FI.csv_read_matrix(gt_file, delim=" ", comment_str="#")
        mat = np.array(raw_mat).astype(float)
        gt_stamps = mat[:, 0]
        xyz = mat[:, 1:4]  # n x 3
        quat = mat[:, 4:]  # n x 4
        # shift 1 column -> w in front column
        quat = np.roll(quat, 1, axis=1)
        gt_abs_poses = TJ.xyz_quat_wxyz_to_se3_poses(xyz, quat)
        return gt_stamps, gt_abs_poses

    def plot_error(self, stamps1, error1, stamps2, error2):
        e1_array = np.array(error1)
        s1_array = np.array(stamps1)
        e2_array = np.array(error2)
        s2_array = np.array(stamps2)

        print np.mean(e1_array)
        print np.mean(e2_array)

        # plot the data
        plt.title('errors')
        plt.xlabel('timestamps')
        plt.ylabel('RPE [m]')
        plt.plot(s1_array, e1_array, 'b')
        plt.plot(s2_array, e2_array, 'g')
        # plt.plot(timestamps, error, 'b.')
        plt.savefig("./gf_error.png")
        plt.show()
        plt.close()


if __name__ == '__main__':
    gt_path = '/media/duyanwei/du/data/Euroc/MAV/MH_01_easy/mav0/cam0/data.tum'
    frame_path = '/home/duyanwei/slam_ws/code/dso/dso/build/MH_01_easy_5/stats_tracking_result.txt'
    frame_path2 = '/home/duyanwei/slam_ws/code/dso/dso/build/MH_01_easy_51/stats_tracking_result.txt'
    ft = EvaluateTracking(gt_path, frame_path2, 1.32146890577)
    # ft2 = EvaluateTracking2(gt_path, frame_path, 1.34082548985, frame_path2, 1.32146890577)

# 1.33998230459
# 0.842022612795
# 0.243913617442
# 0.667879414404
# 1.55858988256
