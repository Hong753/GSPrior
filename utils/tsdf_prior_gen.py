import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.grid_fuser import TSDF_Fuser
import sys
import open3d as o3d


def run_tsdf_fusion(model_path, source_path,sdf_trunc, iteration=-1, voxel_size=0.02, depth_trunc=5.0):
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    parser = ArgumentParser(description="TSDF Fusion parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=iteration, type=int)
    sys.argv = [
        sys.argv[0],
        f"--model_path={model_path}",
        f"--source_path={source_path}",
        f"--iteration={iteration}"
    ]
    args = get_combined_args(parser)
    args.model = model
    args.pipeline = pipeline
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    tsdfFuser = TSDF_Fuser(
        gaussians,
        render,
        pipe,
        bg_color=bg_color,
        voxel_size=voxel_size,
        depth_trunc=depth_trunc,
        sdf_trunc=sdf_trunc
    )
    query_pts,tsdf_vol, vol_origin, voxel_size, mesh = \
        tsdfFuser.reconstruction(scene.getTrainCameras())
    return  query_pts, tsdf_vol, vol_origin, voxel_size,mesh