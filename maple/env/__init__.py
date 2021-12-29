import sys
# 
from .ant import AntEnv
from .humanoid import HumanoidEnv
from .halfcheetah_jump import HalfCheetahEnv as HalfCheetahJumpEnv
from .halfcheetah_vel import HalfCheetahEnv as HalfCheetahVelEnv
from .ant_angle import AntEnv as AngAngleEnv
# import halfcheetah_vel
# 
env_overwrite = {'Ant': AntEnv,'AntAngle': AngAngleEnv,
                 'Humanoid': HumanoidEnv, 'HalfCheetahVel':HalfCheetahVelEnv,
                 'HalfCheetahJump': HalfCheetahJumpEnv}
# 
# sys.modules[__name__] = env_overwrite