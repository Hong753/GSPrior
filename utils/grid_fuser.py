# tsdf_fuser.py
from utils import fusion
import torch
import numpy as np
import math
from tqdm import tqdm
import trimesh
import open3d as o3d


class TSDF_Fuser(object):
    def __init__(self, gaussians, render, pipe,sdf_trunc, bg_color=None, voxel_size=0.02, depth_trunc=5.0):
        """
        TSDF Fusion class for mesh reconstruction.

        Example:
        >>> tsdfFuser = TSDF_Fuser(gaussians, render, pipe)
        >>> query_points, tsdf_vol, mesh_fusion, mesh_o3d, vol_origin, voxel_size, sampled_pts = tsdfFuser.reconstruction(viewpoint_stack)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]

        self.gaussians = gaussians
        self.render_func = render
        self.pipe = pipe
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.voxel_size = voxel_size
        self.depth_trunc = depth_trunc
        self.sdf_trunc =sdf_trunc
        self.depthmaps = [] 

    @staticmethod
    def fov2focal(fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    @staticmethod
    def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    @staticmethod
    def get_extrinsic(viewpoint_cam):
        return torch.tensor(
            TSDF_Fuser.getWorld2View2(
                viewpoint_cam.R,
                viewpoint_cam.T,
                translate=np.array([.0, .0, .0]),
                scale=1.0
            )
        ).cuda()

    @staticmethod
    def get_intrinsic(viewpoint_cam):
        FovY = viewpoint_cam.FoVy
        FovX = viewpoint_cam.FoVx
        H = viewpoint_cam.image_height
        W = viewpoint_cam.image_width
        fx = W / (2 * math.tan(FovX / 2))
        fy = H / (2 * math.tan(FovY / 2))
        intrins = torch.tensor(
            [[fx, 0., W/2.],
             [0., fy, H/2.],
             [0., 0., 1.0]]
        ).float().cuda()
        return intrins

    @staticmethod
    def post_process_mesh_robust(mesh: trimesh.Trimesh,
                                 cluster_to_keep=1,
                                 min_faces=50,
                                 min_area=1e-4,
                                 min_volume=1e-6,
                                 max_center_distance=None,
                                 verbose=True):
        print("========== Mesh Post-Processing ==========")
        print(f"Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        components = mesh.split(only_watertight=False)
        global_center = mesh.centroid
        filtered = []
        for m in components:
            if len(m.faces) < min_faces:
                continue
            if m.area < min_area:
                continue
            if m.is_watertight and abs(m.volume) < min_volume:
                continue
            if max_center_distance is not None:
                dist = np.linalg.norm(m.centroid - global_center)
                if dist > max_center_distance:
                    continue
            filtered.append(m)

        if len(filtered) == 0:
            if verbose:
                print("Warning: all clusters filtered out, keeping original mesh.")
            return mesh.copy()

        filtered = sorted(filtered, key=lambda m: len(m.faces), reverse=True)
        kept = filtered[:cluster_to_keep]

        mesh_processed = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)
        mesh_processed.remove_duplicate_faces()
        mesh_processed.remove_degenerate_faces()
        mesh_processed.remove_unreferenced_vertices()

        print(f"Processed mesh: {len(mesh_processed.vertices)} vertices, "
              f"{len(mesh_processed.faces)} faces")
        return mesh_processed

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, use_depth_filter=True, num_samples=200000):
        self.viewpoint_stack = viewpoint_stack
        K = self.get_intrinsic(self.viewpoint_stack[0])

        vol_bnds = np.zeros((3, 2))

        for viewpoint_cam in tqdm(self.viewpoint_stack, desc="Estimate bounds"):
            pose_w2c = self.get_extrinsic(viewpoint_cam)
            pose_c2w = np.linalg.inv(pose_w2c.detach().cpu().numpy())

            out = self.render_func(viewpoint_cam,
                                   self.gaussians,
                                   pipe=self.pipe,
                                   bg_color=self.background)
            depth_tsdf = out["plane_depth"].squeeze().clone()
            depth_tsdf[depth_tsdf > self.depth_trunc] = 0

            if use_depth_filter and "depth_normal" in out:
                view_dir = torch.nn.functional.normalize(viewpoint_cam.get_rays(), p=2, dim=-1)
                depth_normal = out["depth_normal"].permute(1, 2, 0)
                depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
                dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
                angle = torch.acos(dot)
                mask = angle > (80.0 / 180.0 * math.pi)
                depth_tsdf[mask] = 0

            view_frust_pts = fusion.get_view_frustum(
                depth_tsdf.detach().cpu().numpy(),
                K.detach().cpu().numpy(),
                pose_c2w
            )
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        vol_bnds = np.array([
            [-1, 1],  # X
            [-1, 1],  # Y
            [-1, 1],  # Z
        ], dtype=np.float32)#scale to 1m by defalut, change comment it with vol_bnds by depth_tsdf
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=self.voxel_size,sdf_trunc=self.sdf_trunc)
        print("Estimated volume bounds:\n", vol_bnds)

        # === Pass 2: TSDF Fusion ===
        for viewpoint_cam in tqdm(self.viewpoint_stack, desc="TSDF Fusion progress"):
            out = self.render_func(viewpoint_cam,
                                   self.gaussians,
                                   pipe=self.pipe,
                                   bg_color=self.background)

            pose_w2c = self.get_extrinsic(viewpoint_cam)
            pose_c2w = np.linalg.inv(pose_w2c.detach().cpu().numpy())

            img_cur = out['render'].permute(1, 2, 0).detach().cpu().numpy()
            depth_t = out['plane_depth'].squeeze().clone()
            depth_t[depth_t > self.depth_trunc] = 0

            if use_depth_filter and "depth_normal" in out:
                view_dir = torch.nn.functional.normalize(viewpoint_cam.get_rays(), p=2, dim=-1)
                depth_normal = out["depth_normal"].permute(1, 2, 0)
                depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
                dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
                angle = torch.acos(dot)
                mask = angle > (80.0 / 180.0 * math.pi)
                depth_t[mask] = 0

            depth_np = depth_t.detach().cpu().numpy()

            # Graphdeco TSDF
            tsdf_vol.integrate(
                img_cur,
                depth_np,
                K.detach().cpu().numpy(),
                pose_c2w,
                obs_weight=1.0
            )
        query_points, tsdf_vol, mesh_fusion, vol_origin, voxel_size = tsdf_vol.get_mesh()
        print("TSDF Fusion Done (fusion + Open3D).")
        return query_points,tsdf_vol, vol_origin, voxel_size,mesh_fusion
