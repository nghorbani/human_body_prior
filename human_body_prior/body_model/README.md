# VPoser PoZ Space for Body Models
The original body pose space of [SMPL](http://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) 
(jointly refered to SMPL) are not bounded to natural human pose space. That means you can put a vector value 
as the pose of a SMPL body model and get a messed up body, that might not even look like a human.
To address this you can replace the original pose space of SMPL with VPoser's latent space that corresponds to natural human pose space.
This means if you sample a 32 dimentional random vector from a Normal distribution and pass it as your pose you should get 
a viable human pose. This representation of pose is fully differentiable and can be used in an end-to-end pipeline.

An example use case is an expressive 2D-to-3D lifting of a human image done in the paper  [Expressive Body Capture:
3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/)
 
To load a SMPL with this pose representation you can use the following snippet:
```python
from human_body_prior.body_model.body_model import BodyModelWithPoser

bm_path = 'PATH_TO_SMPLX_model.npz' # obtain from https://smpl-x.is.tue.mpg.de/downloads

smpl_exp_dir = 'TRAINED_MODEL_DIRECTORY' # directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads

bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=smpl_exp_dir).to('cuda')

```
In the above snippet, *bm* is a BodyModel isntance where the body pose is not represented by body_pose anymore, 
but with body_poZ that is the latent space of respective VPoser model. So in any optiization task that involves a human body
you can use bm.body_poZ as the free variable and put a L2 norm on this vriable to keep it close to the mean of Normal.

Furthermore, you can randomize the body_poZ and visualise it using *trimesh*:  

```python
import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c

bm.randomize_pose()

vertices = c2c(bm.forward().v)[0]
faces = c2c(bm.f)

mesh = trimesh.base.Trimesh(vertices, faces).show()
```

# Disentangling Self-Intersecting Novel Poses
Using VPoser you can sample new poses from the human body pose distribution provided by VPoser. 
These poses usually don't have artifacts like interpenetration, e.g. self-intersecting vertices.
However, interpentrating samples might still happen probably due to artifacts in the training set of VPoser, e.g. [AMASS](https://amass.is.tue.mpg.de/). 
In these cases one can use a tool like [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) to disentangle the interpenetrating parts.
We have integrated the code for doing this inside the *BodyModelWithPoser* class for convenience:
```python

bm.randomize_pose()

untangle = bm.untagnle_interpenetrations()
untangle()

vertices = c2c(bm.forward().v)[0]
faces = c2c(bm.f)

mesh = trimesh.base.Trimesh(vertices, faces).show()
```
