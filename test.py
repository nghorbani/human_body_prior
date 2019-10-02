import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModelWithPoser
import torch


def BhattacharyyaDistance(dist1 :torch.tensor, dist2:torch.tensor):
    s1s , s2s , m1 , m2 = dist1.var()**2 , dist2.var()**2 , dist1.mean() , dist2.mean()

    return (torch.log(((s1s/s2s) + (s2s/s1s) + (2))/4.0) + ((m1 - m2)**2)/(4.0 * (s1s + s2s)))/4.0

if __name__ == "__main__":
    bm_path = 'D:\\VCL\\Users\\vlad\\Misc\\vposer_v1_0 (1)\\model.pkl' # obtain from https://smpl-x.is.tue.mpg.de/downloads

    smpl_exp_dir = 'D:\\VCL\\Users\\vlad\\Misc\\vposer_v1_0 (1)' # directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads

    import numpy
    # q = numpy.load(bm_path,allow_pickle = True)
    # numpy.save("D:\\VCL\\Users\\vlad\\Misc\\vposer_v1_0 (1)\\model.npz",q)
    smpl_dict = numpy.load(bm_path, allow_pickle = True)
    bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=smpl_exp_dir).to('cuda')

    bm.randomize_pose()
    q = bm.forward()
    sampler = torch.distributions.Normal(0,1)
    sampler2 = torch.distributions.Beta(10,20)
    q_ = sampler.sample(q.poZ_body.shape)
    p_ = sampler2.sample(q.poZ_body.shape)
    #z = torch.distributions.kl_divergence(q.poZ_body.cpu().cpu(), q_)
    z = BhattacharyyaDistance(p_ , q_)

    vertices = c2c(bm.forward().v)[0]
    faces = c2c(bm.f)

    mesh = trimesh.base.Trimesh(vertices, faces).show()