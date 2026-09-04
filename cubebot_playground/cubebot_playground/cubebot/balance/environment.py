from __future__ import annotations
from pathlib import Path
from typing import Optional
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from . import constants
from .config import default_config
from .math_utils import quat_rotate_inverse
from .observations import build_observation
from .rewards import compute_rewards

class CubebotBalance(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(), config_overrides: Optional[dict] = None):
        super().__init__(config, config_overrides=config_overrides)
        self.episode_length = self._config.episode_length
        self._xml_path = str(Path(__file__).with_name("scene_balance.xml"))
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        if self._mj_model.nu != constants.ACTION_SIZE:
            raise ValueError(f"Expected 12 actuators, found {self._mj_model.nu}")
        rid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root_freejoint")
        self._root_qpos_adr = int(self._mj_model.jnt_qposadr[rid]); self._root_dof_adr = int(self._mj_model.jnt_dofadr[rid])
        qaddrs=[]; daddrs=[]; defaults=[]
        for name in constants.JOINT_NAMES:
            jid=mujoco.mj_name2id(self._mj_model,mujoco.mjtObj.mjOBJ_JOINT,name)
            qaddrs.append(int(self._mj_model.jnt_qposadr[jid])); daddrs.append(int(self._mj_model.jnt_dofadr[jid])); defaults.append(constants.DEFAULT_JOINT_POS[name])
        self._joint_qpos_adrs=jp.array(qaddrs,dtype=jp.int32); self._joint_dof_adrs=jp.array(daddrs,dtype=jp.int32); self._default_joint_pos=jp.array(defaults)
        ctrl=[]
        for aid in range(self._mj_model.nu):
            jid=int(self._mj_model.actuator_trnid[aid,0]); jname=mujoco.mj_id2name(self._mj_model,mujoco.mjtObj.mjOBJ_JOINT,jid); ctrl.append(constants.DEFAULT_JOINT_POS[jname])
        self._default_ctrl=jp.array(ctrl)

    def reset(self, rng):
        rng_height, rng_tilt = jax.random.split(rng)
        desired_height=jax.random.uniform(rng_height,minval=self._config.height_min,maxval=self._config.height_max)
        qpos=jp.array(self._mj_model.qpos0); qvel=jp.zeros(self._mj_model.nv)
        qpos=qpos.at[self._joint_qpos_adrs].set(self._default_joint_pos)
        qpos=qpos.at[self._root_qpos_adr:self._root_qpos_adr+3].set(jp.array([0.0,0.0,desired_height]))
        rp=jax.random.uniform(rng_tilt,shape=(2,),minval=-0.05,maxval=0.05); roll,pitch=rp[0],rp[1]
        cr,sr=jp.cos(roll/2),jp.sin(roll/2); cp,sp=jp.cos(pitch/2),jp.sin(pitch/2)
        quat=jp.array([cp*cr,cp*sr,sp*cr,-sp*sr])
        qpos=qpos.at[self._root_qpos_adr+3:self._root_qpos_adr+7].set(quat)
        ctrl=self._default_ctrl
        data=mjx_env.make_data(self._mj_model,qpos=qpos,qvel=qvel,ctrl=ctrl)
        info={"desired_height":desired_height,"previous_action":jp.zeros(12),"servo_command":ctrl,"step_count":jp.array(0,dtype=jp.int32)}
        obs=self._get_obs(data,info)
        metrics={"reward/upright":jp.array(0.0),"reward/height":jp.array(0.0),"penalty/ang_vel":jp.array(0.0),"penalty/joint_vel":jp.array(0.0),"penalty/pose":jp.array(0.0),"penalty/action_rate":jp.array(0.0),"body_height":data.qpos[self._root_qpos_adr+2],"desired_height":desired_height}
        return mjx_env.State(data,obs,jp.array(0.0),jp.array(0.0),metrics,info)

    def step(self,state,action):
        action=jp.clip(action,-1.0,1.0)
        desired_ctrl=self._default_ctrl+self._config.action_scale*action
        max_delta=self._config.max_servo_speed*self._config.ctrl_dt
        servo_command=state.info["servo_command"]+jp.clip(desired_ctrl-state.info["servo_command"],-max_delta,max_delta)
        data=mjx_env.step(self.mjx_model,state.data,servo_command,self.n_substeps)
        rq=data.qpos[self._root_qpos_adr+3:self._root_qpos_adr+7]; rw=data.qvel[self._root_dof_adr+3:self._root_dof_adr+6]
        bav=quat_rotate_inverse(rq,rw); pg=quat_rotate_inverse(rq,jp.array([0.0,0.0,-1.0]))
        jp0=data.qpos[self._joint_qpos_adrs]; jv=data.qvel[self._joint_dof_adrs]; h=data.qpos[self._root_qpos_adr+2]; dh=state.info["desired_height"]
        reward,parts=compute_rewards(projected_gravity=pg,body_height=h,desired_height=dh,body_ang_vel=bav,joint_pos=jp0,default_joint_pos=self._default_joint_pos,joint_vel=jv,action=action,previous_action=state.info["previous_action"],config=self._config)
        count=state.info["step_count"]+1
        fallen=jp.logical_or(h<self._config.min_body_height,pg[2]>self._config.min_projected_gravity_z); timeout=count>=self._config.episode_length; done=jp.logical_or(fallen,timeout).astype(jp.float32)
        info={**state.info,"previous_action":action,"servo_command":servo_command,"step_count":count}
        obs=self._get_obs(data,info)
        # Preserve keys injected by wrappers (for example, EvalWrapper adds
        # ``reward``).  JAX scan requires the carry pytree to keep exactly the
        # same structure between reset and every step.
        metrics={**state.metrics,"reward/upright":parts["upright"],"reward/height":parts["height"],"penalty/ang_vel":parts["ang_vel_penalty"],"penalty/joint_vel":parts["joint_vel_penalty"],"penalty/pose":parts["pose_penalty"],"penalty/action_rate":parts["action_rate_penalty"],"body_height":h,"desired_height":dh}
        return mjx_env.State(data,obs,reward,done,metrics,info)

    def _get_obs(self,data,info):
        rq=data.qpos[self._root_qpos_adr+3:self._root_qpos_adr+7]; rw=data.qvel[self._root_dof_adr+3:self._root_dof_adr+6]
        return build_observation(root_quat=rq,root_ang_vel_world=rw,servo_command=info["servo_command"],default_servo_command=self._default_ctrl,action_scale=self._config.action_scale,desired_height=info["desired_height"],height_min=self._config.height_min,height_max=self._config.height_max,ang_vel_scale=self._config.obs_ang_vel_scale)
    @property
    def observation_size(self): return constants.OBSERVATION_SIZE
    @property
    def action_size(self): return constants.ACTION_SIZE
    @property
    def xml_path(self): return self._xml_path
    @property
    def mj_model(self): return self._mj_model
    @property
    def mjx_model(self): return self._mjx_model
