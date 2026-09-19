"""Phase2: fixed paired seeds, original7D actions, proximity+dwell, all actuated-joint loads.

This is a separate protocol: no Phase1 action deadband or Phase1 marker success.
Success is geometric proximity at0.10m;0.05m is a secondary threshold, not safety.
Raw orientation is saved without asserting an uncalibrated docking-frame error.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W));sys.path.insert(0,'WITHHELD_LOCAL_PATH')
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch,torchvision,mujoco
from PIL import Image
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from mujoco_env.y_env_berthing_v2 import BerthingEnvV2
from scripts.revision_primary_utils_20260911 import reset_environment_exact,seed_everything,runtime_metadata,sha256_file
from scripts.revision_render_audit_20260910 import configure_render,audit_episode

def save(path,data):
 p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
def array_sha(a):return hashlib.sha256(np.asarray(a,dtype=np.float32).tobytes()).hexdigest()

class Phase2Logger:
 def __init__(self,p):
  self.p=p;self.m=p.env.model;self.d=p.env.data
  self.jids=[i for i in range(self.m.njnt) if self.m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
  self.names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in self.jids]
  assert self.names==['waist','shoulder','elbow','forearm_roll','wrist_angle'], self.names
  self.nj=len(self.names)
  self.qa=np.array([self.m.jnt_qposadr[i] for i in self.jids]);self.da=np.array([self.m.jnt_dofadr[i] for i in self.jids])
  self.ids=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['iss_docking_marker','cygnus_docking_marker']];assert min(self.ids)>=0
  self.start=float(self.d.time);self.last=self.start;self.n=0
  self.signed=np.zeros(self.nj);self.positive=np.zeros(self.nj);self.absolute=np.zeros(self.nj);self.peak=np.zeros(self.nj);self.peak_tau=np.zeros(self.nj);self.peak_vel=np.zeros(self.nj)
  self.margin=np.full(self.nj,np.inf);self.force_peak=0.;self.impulse=0.;self.docking_peak=0.;self.docking_impulse=0.
  self.first={};self.after_n=0;self.after_within={.1:0,.05:0};self.after_contact=0;self.min_distance=float('inf')
 def geometry(self):
  a,b=self.ids;pa=self.d.xpos[a].copy();pb=self.d.xpos[b].copy();ra=self.d.xmat[a].reshape(3,3).copy();rb=self.d.xmat[b].reshape(3,3).copy()
  va=np.zeros(6);vb=np.zeros(6)
  for i,v in [(a,va),(b,vb)]:mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_XBODY,i,v,0)
  relative=ra.T@(pb-pa)
  return {'distance_m':float(np.linalg.norm(relative)),'relative_position_iss_axes_m':relative.tolist(),'iss_position_world_m':pa.tolist(),'cygnus_position_world_m':pb.tolist(),'relative_rotation_iss_to_cygnus':(ra.T@rb).tolist(),'iss_quaternion_wxyz':self.d.xquat[a].tolist(),'cygnus_quaternion_wxyz':self.d.xquat[b].tolist(),'relative_linear_velocity_world_m_s':(vb[3:]-va[3:]).tolist(),'relative_angular_velocity_world_rad_s':(vb[:3]-va[:3]).tolist(),'relative_speed_m_s':float(np.linalg.norm(vb[3:]-va[3:])),'relative_angular_speed_rad_s':float(np.linalg.norm(vb[:3]-va[:3]))}
 def contacts(self):
  rows=[]
  for i in range(self.d.ncon):
   c=self.d.contact[i];f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,i,f)
   a,b=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])
   rows.append({'bodies':[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,k) for k in [a,b]],'force_n':float(np.linalg.norm(f[:3])),'docking_marker_pair':set([a,b])==set(self.ids)})
  return rows
 def observe(self):
  dt=float(self.m.opt.timestep);now=float(self.d.time);assert abs(now-self.last-dt)<1e-8,'Missing physics step'
  self.last=now;self.n+=1;q=self.d.qpos[self.qa];v=self.d.qvel[self.da];tau=self.d.qfrc_actuator[self.da];power=tau*v
  self.signed+=power*dt;self.positive+=np.maximum(power,0)*dt;self.absolute+=abs(power)*dt;self.peak=np.maximum(self.peak,abs(power));self.peak_tau=np.maximum(self.peak_tau,abs(tau));self.peak_vel=np.maximum(self.peak_vel,abs(v))
  for k,j in enumerate(self.jids):
   if self.m.jnt_limited[j]:self.margin[k]=min(self.margin[k],q[k]-self.m.jnt_range[j,0],self.m.jnt_range[j,1]-q[k])
  contacts=self.contacts();forces=[r['force_n'] for r in contacts];df=[r['force_n'] for r in contacts if r['docking_marker_pair']]
  self.force_peak=max([self.force_peak]+forces);self.impulse+=sum(forces)*dt;self.docking_peak=max([self.docking_peak]+df);self.docking_impulse+=sum(df)*dt
  dist=float(np.linalg.norm(self.d.xpos[self.ids[0]]-self.d.xpos[self.ids[1]]));self.min_distance=min(self.min_distance,dist)
  for t in [.1,.05]:
   if dist<t and str(t) not in self.first:self.first[str(t)]={'time_s':now,'geometry':self.geometry(),'contacts':contacts}
  if '0.1' in self.first:
   self.after_n+=1;self.after_contact+=int(bool(df))
   for t in [.1,.05]:self.after_within[t]+=int(dist<t)
 def row(self,step,action):
  return {'policy_step':step,'sim_time_s':float(self.d.time),'action':np.asarray(action).tolist(),**self.geometry(),'joint_position_rad':self.d.qpos[self.qa].tolist(),'joint_velocity_rad_s':self.d.qvel[self.da].tolist(),'joint_actuator_torque_nm':self.d.qfrc_actuator[self.da].tolist(),'joint_mechanical_power_w':(self.d.qvel[self.da]*self.d.qfrc_actuator[self.da]).tolist(),'contacts':self.contacts()}
 def summary(self):
  return {'joint_names':self.names,'physics_steps':self.n,'physics_integrated_duration_s':self.n*float(self.m.opt.timestep),'per_joint_signed_work_j':self.signed.tolist(),'per_joint_positive_work_j':self.positive.tolist(),'per_joint_absolute_work_j':self.absolute.tolist(),'per_joint_peak_absolute_power_w':self.peak.tolist(),'per_joint_peak_absolute_torque_nm':self.peak_tau.tolist(),'per_joint_peak_absolute_velocity_rad_s':self.peak_vel.tolist(),'per_joint_min_limit_margin_rad':[float(v) if np.isfinite(v) else None for v in self.margin],'peak_single_contact_force_n':self.force_peak,'contact_force_magnitude_integral_n_s':self.impulse,'peak_docking_marker_contact_force_n':self.docking_peak,'docking_marker_force_magnitude_integral_n_s':self.docking_impulse,'first_threshold_crossings':self.first,'minimum_distance_m':self.min_distance,'post_threshold_physics_samples':self.after_n,'post_threshold_fraction_within_0_10m':self.after_within[.1]/self.after_n if self.after_n else None,'post_threshold_fraction_within_0_05m':self.after_within[.05]/self.after_n if self.after_n else None,'post_threshold_marker_contact_fraction':self.after_contact/self.after_n if self.after_n else None,'terminal':self.geometry(),'electrical_energy_j':None,'secure_capture_success':None}

def episode(p,policy,device,transform,args,pair,out,index):
 seed_everything(pair['policy_seed']);reset_environment_exact(p,pair['env_seed']);policy.reset();policy.eval()
 render=audit_episode(p,'fixed',out,index);logger=Phase2Logger(p)
 assert logger.geometry()['distance_m']>.10,'Initial scene already meets proximity criterion'
 original=p.step_env
 def tracked():original();logger.observe()
 p.step_env=tracked;pol=0;actions=[];last=np.zeros(7,dtype=np.float32);wall=time.monotonic();p.grab_image();sub=max(1,int(round(p.env.HZ/20)))
 trace=out/f'attempt_{index:04d}_trace.jsonl';reason='max_steps'
 try:
  with trace.open('x') as f:
   for _ in range(args.max_steps*sub+5000):
    first=logger.first.get('0.1')
    if first and float(p.env.data.time)-first['time_s']>=args.dwell:reason='threshold_and_followup';break
    if p.env.loop_every(HZ=20):
     if pol>=args.max_steps:break
     state=p.get_ee_pose();joints=p.get_joint_state();top,front,head,side=p.grab_image()
     data={'observation.state':torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0),'observation.joint_state':torch.tensor(joints,dtype=torch.float32,device=device).unsqueeze(0),'task':['Dock space ship to station'],'timestamp':torch.tensor([[pol/20]],device=device)}
     for key,rgb in [('topview',top),('frontview',front),('sideview',side)]:data['observation.'+key]=transform(Image.fromarray(rgb).resize((256,256))).unsqueeze(0).to(device)
     with torch.inference_mode():last=policy.select_action(data).squeeze(0).cpu().numpy()
     assert last.shape==(7,) and np.isfinite(last).all();actions.append(last.copy());p.step(last);pol+=1;f.write(json.dumps(logger.row(pol,last),allow_nan=False)+'\n')
     if pol%20==0:
      f.flush();save(out/'progress.json',{'status':'evaluating','attempt':index,'policy_steps':pol,'elapsed_s':time.monotonic()-wall,'updated_at':datetime.now().astimezone().isoformat()})
    else:p.step_env()
   f.write(json.dumps(logger.row(pol,last),allow_nan=False)+'\n');f.flush()
 finally:p.step_env=original
 assert abs(logger.n*float(p.env.model.opt.timestep)-(float(p.env.data.time)-logger.start))<1e-7
 np.save(out/f'attempt_{index:04d}_actions.npy',np.asarray(actions,dtype=np.float32))
 r={'attempt':index,**pair,'success':bool('0.1' in logger.first),'secondary_reached_0_05m':bool('0.05' in logger.first),'reason':reason,'post_threshold_followup_complete':reason=='threshold_and_followup','policy_steps':pol,'wall_s':time.monotonic()-wall,'raw_action_sha256':array_sha(actions),'render_audit':render,'trace_sha256':sha256_file(trace),**logger.summary()}
 save(out/f'attempt_{index:04d}_result.json',r);print(json.dumps({'attempt':index,'success':r['success'],'steps':pol,'wall_s':r['wall_s']}),flush=True);return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--max-steps',type=int,default=20000);ap.add_argument('--dwell',type=float,default=2.0);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 os.chdir(W);out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume);pairs=json.loads(Path(a.manifest).read_text())['pairs'];ck=Path(a.ckpt)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('highest');seed_everything(0)
 device=torch.device('cuda');policy=DiffusionPolicy.from_pretrained(str(ck)).to(device);policy.eval();policy.diffusion.num_inference_steps=10
 assert list(policy.config.image_features)==['observation.topview','observation.frontview','observation.sideview'];assert list(policy.config.output_features['action'].shape)==[7]
 p=BerthingEnvV2(headless=True,seed=0,init_progress=True,dock_dist_threshold=.10);render=configure_render(p,'fixed');assert 'llvmpipe' in render['gl_renderer'].lower()
 proof={'protocol_version':'phase2_paired_safety_20260912','evaluator_sha256':sha256_file(Path(__file__)),'environment_sha256':sha256_file(W/'mujoco_env/y_env_berthing_v2.py'),'xml_sha256':sha256_file(W/'asset/berthing/berthing_scene_v2.xml'),'checkpoint_sha256':sha256_file(ck/'model.safetensors'),'config_sha256':sha256_file(ck/'config.json'),'seed_manifest_sha256':sha256_file(Path(a.manifest)),'max_policy_steps':a.max_steps,'dwell_seconds':a.dwell,'hz':20,'inference_steps':10,'action_postprocessing':'none; original seven-dimensional action','render_configuration':render,'runtime':runtime_metadata(),'primary_success':'marker distance<0.10m; geometric only','secondary_threshold':'0.05m reached within the same fixed observation horizon','scene_randomization':'lights only; initial geometry fixed','orientation_note':'raw relative frames; desired mating rotation not assumed','energy_note':'mechanical actuator work, not electrical input','training_comparison_note':'legacy and matched-budget checkpoints kept separate'}
 if a.resume:assert json.loads((out/'protocol.json').read_text())==proof
 else:save(out/'protocol.json',proof)
 rows=[]
 try:
  for i,pair in enumerate(pairs,1):
   saved=out/f'attempt_{i:04d}_result.json'
   if saved.exists():
    r=json.loads(saved.read_text());assert all(r[k]==pair[k] for k in ['env_seed','policy_seed']);assert sha256_file(out/f'attempt_{i:04d}_trace.jsonl')==r['trace_sha256'];rows.append(r);continue
   assert not (out/f'attempt_{i:04d}_trace.jsonl').exists(),'Partial episode preserved; no automatic overwrite'
   rows.append(episode(p,policy,device,torchvision.transforms.ToTensor(),a,pair,out,i))
 finally:p.env.close_viewer()
 save(out/'summary.json',{**proof,'n_attempts':len(rows),'n_success':sum(r['success'] for r in rows),'per_attempt':rows});save(out/'progress.json',{'status':'complete','n':len(rows),'updated_at':datetime.now().astimezone().isoformat()})
if __name__=='__main__':main()
