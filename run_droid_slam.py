import sys

from tqdm import tqdm
from pathlib import Path
from typing import List, Any
import numpy as np
import torch
import lietorch
import cv2
import open3d as o3d
import os
import glob 
import time
import argparse
import submitit

from torch.multiprocessing import Process
import torch.nn.functional as F

def image_stream(imagedir, calib, stride):
    """ image generator """

    # calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("{}/images.npy".format(reconstruction_path), images)
    np.save("{}/disps.npy".format(reconstruction_path), disps)
    np.save("{}/poses.npy".format(reconstruction_path), poses)
    np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=Path, help="path to image directory")
    parser.add_argument("--outputdir", type=Path, help="path to save reconstruction")
    parser.add_argument("--calib", type=Path, help="path to calibration file")
    parser.add_argument("--calib_method", help="ctrlc,colmap", default="ctrlc")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--global_ba", action="store_true")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--slurm-partition', default='devlab')
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    # need high resolution depths
    if args.outputdir is not None:
        args.upsample = True

    img_dirs = [p for p in args.imagedir.iterdir() if p.is_dir]
    out_dirs = [args.outputdir / p.name for p in img_dirs]

    if args.num_workers == 0:
        print(args.calib_method)
        for imagedir, outputdir in zip(img_dirs, out_dirs):
            reconstruct(args, imagedir, outputdir)
    else:
        run_on_slurm(args, img_dirs, out_dirs)

def split_tasks(tasks: List[Any], num_workers: int) -> List[List[Any]]:
    """ Split a list of tasks up into equal-ish sized chunks """
    # Figure out how many tasks each worker should handle. We want to split as
    # evenly as possible, so the difference in number of tasks per worker
    # varies by at most one. We first split evenly (rounding down) then go back
    # and assign one extra task per worker as needed.
    tasks_per_worker = [len(tasks) // num_workers for _ in range(num_workers)]
    for i in range(num_workers):
        if sum(tasks_per_worker) == len(tasks):
            break
        tasks_per_worker[i] += 1
    assert sum(tasks_per_worker) == len(tasks)

    # Now actually split tasks
    sharded_tasks = [[]]
    for task in tasks:
        worker_idx = len(sharded_tasks) - 1
        if len(sharded_tasks[worker_idx]) == tasks_per_worker[worker_idx]:
            sharded_tasks.append([])
        sharded_tasks[-1].append(task)
    for num, shard in zip(tasks_per_worker, sharded_tasks):
        assert num == len(shard)

    return sharded_tasks


def run_on_slurm(args, img_dirs: List[Path], out_dirs: List[Path]):
    tasks = [(args, i, o) for i, o in zip(img_dirs, out_dirs)]
    sharded_tasks = split_tasks(tasks, args.num_workers)

    slurm_folder = args.outputdir / 'slurm'
    executor = submitit.AutoExecutor(folder=slurm_folder)
    executor_params = {
        'gpus_per_node': 1,
        'tasks_per_node': 1,
        'mem_gb': 64,
        'cpus_per_task': 8,
        'nodes': 1,
        'timeout_min': 60 * 48,
        'slurm_partition': args.slurm_partition,
    }
    executor.update_parameters(**executor_params)
    executor.update_parameters(slurm_srun_args=["-vv", "--cpu-bind", "none"])
    jobs = []
    with executor.batch():
        for shard in sharded_tasks:
            job = executor.submit(TaskRunner(shard))
            jobs.append(job)
    for job in jobs:
        print(f'Submitted job {job.job_id}')
    print(f'Job dir: {slurm_folder}')


class TaskRunner:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self):
        num_tasks = len(self.tasks)
        for i, (args, img_dir, out_dir) in enumerate(self.tasks):
            print(f'Running reconstruction {i + 1} / {num_tasks}')
            reconstruct(args, img_dir, out_dir)

def parse_calib(calib_txt):
    f = open(calib_txt,"r")
    lines = f.readlines()

    fxs = []
    for line_id, line in enumerate(lines):
        if line_id == 2:
            num_cameras = int(line.split("# Number of cameras: ")[1])
        elif line_id > 2:
            entry = line.split(" ")
            fxs.append(float(entry[4]))
    assert(num_cameras == len(fxs))
    fx = np.median(np.array(fxs))
    cx, cy = float(entry[5]), float(entry[6])
    intrinsics = np.array([fx, fx, cx, cy])
    print("Camera Intrinsics: ", intrinsics)
    return intrinsics

def reconstruct(args, imagedir, outputdir):
    if 'droid_slam' not in sys.path:
        sys.path.append('droid_slam')
    from droid import Droid
    from visualization_headless import droid_visualization
    print("Processing: ", imagedir)
    output_file = "{}/output.ply".format(outputdir)
    if os.path.isfile(output_file):
        print("Output file already exists! Skipping.")
        return
    calibfile = "{}/{}/cameras.txt".format(args.calib, imagedir.name)
    if args.calib_method == "ctrlc":
        calibfile = "{}/{}.npy".format(args.calib, imagedir.name)
    print("Calib: ", calibfile)
    if not os.path.isfile(calibfile):
        print("Camera file not found! Skipping.")
        return
    droid = None
    tstamps = []
    calib = np.load(calibfile) if args.calib_method == "ctrlc" else parse_calib(calibfile)
    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib, args.stride)):
        if t < args.t0:
            continue

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    if args.global_ba:
        traj_est = droid.terminate(image_stream(imagedir, calib, args.stride))

    if outputdir is not None:
        try:
            save_reconstruction(droid, outputdir)
            pcd = droid_visualization(droid.video)
            o3d.io.write_point_cloud(output_file, pcd)
        except:
            print("Failed to save reconstructions!")
            return

if __name__ == '__main__':
    main()
