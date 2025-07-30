import torch
from arti_splatfacto.scene_3d import Scene3D
from arti_splatfacto.obj_3d_seg import Object3DSeg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_gaussians(before, after, title="Articulated Gaussians"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(before[:, 0].cpu(), before[:, 1].cpu(), before[:, 2].cpu(), c='r', label='Pre')
    ax.scatter(after[:, 0].cpu(), after[:, 1].cpu(), after[:, 2].cpu(), c='g', label='Post')
    ax.set_title(title)
    ax.legend()
    plt.show()


def main():
    # Load object segment with articulation info
    obj_seg = Object3DSeg.read_from_file("data/gs_t/obj3Dseg0_updated.pt", device="cuda")

    # Create dummy Gaussians in a tight cube around the pivot
    N = 100
    torch.manual_seed(0)
    pre_means = obj_seg.joint_pivot[None, :] + 0.1 * torch.randn(N, 3).cuda()
    pre_quats = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cuda").repeat(N, 1)

    # Create Scene3D
    scene = Scene3D(device="cuda")
    mask = torch.ones(N, dtype=torch.bool, device="cuda")
    scene.add_object(obj_id=0, obj_seg=obj_seg, gauss_mask=mask)

    # Apply articulation (uses joint_angle from obj_seg)
    out = scene.apply_articulations(pre_means, pre_quats)

    # Print difference or visualize
    print("Before vs After (first 5):")
    for i in range(5):
        print(f"{i}: pre = {pre_means[i].cpu().numpy()}, post = {out['means'][i].cpu().numpy()}")

    # Optional visualization
    plot_gaussians(pre_means, out['means'], title="Pre vs Post Gaussian Means (Articulation)")


if __name__ == "__main__":
    main()
