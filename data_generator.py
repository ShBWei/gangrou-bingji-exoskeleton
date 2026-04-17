"""
Biomimetic Spine Data Generator
Simulates Cervical-Thoracic-Lumbar-Sacral motion with soft-rigid coupling
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class BiomimeticSpineDataset(Dataset):
    """
    Generates synthetic data matching the project book:
    - 4 segments: Cervical, Thoracic, Lumbar, Sacral
    - 3 DOF per segment: Flexion/Extension, Lateral Bend, Rotation
    - Soft-Rigid coupling states based on load/activity
    """
    def __init__(self, num_samples=2000, seq_len=40, pred_len=20, seed=42):
        np.random.seed(seed)
        self.seq_len = seq_len      # 400ms @ 100Hz
        self.pred_len = pred_len    # 200ms prediction
        self.samples = []
        
        for _ in range(num_samples):
            # Time vector
            t = np.linspace(0, 0.6, 60)  # 0.6s total for windowing
            
            # Base motion: Lumbar (waist) drives whole spine
            # Simulates "动态重心感应" from project book
            freq = np.random.uniform(0.5, 1.5)  # Walking/bending frequency
            phase = np.random.uniform(0, 2*np.pi)
            
            # 1. Lumbar (L4-L5): Main load bearing, flexion/extension dominant
            lumbar_flex = np.sin(2*np.pi*freq*t + phase) * 25  # ±25° flexion
            lumbar_bend = np.cos(2*np.pi*freq*t + phase) * 10  # ±10° lateral
            lumbar_rot = np.sin(2*np.pi*freq*t*0.5 + phase) * 15  # ±15° rotation
            
            # 2. Thoracic (T1-T12): Coupled to lumbar, smaller amplitude
            thoracic_flex = lumbar_flex * 0.6 + np.random.normal(0, 2, len(t))
            thoracic_bend = lumbar_bend * 0.7
            thoracic_rot = lumbar_rot * 0.5
            
            # 3. Cervical (C1-C7): Head compensation, inverted phase often
            cervical_flex = thoracic_flex * 0.5 + np.sin(2*np.pi*freq*t + phase + np.pi) * 10
            cervical_bend = thoracic_bend * 0.4
            cervical_rot = thoracic_rot * 0.8 + np.random.normal(0, 3, len(t))
            
            # 4. Sacral (Pelvis): Reference base, minimal motion
            sacral_flex = lumbar_flex * 0.2
            sacral_bend = lumbar_bend * 0.15
            sacral_rot = lumbar_rot * 0.1
            
            # Stack angles: [time, 4_segments, 3_dof]
            angles = np.stack([
                np.stack([cervical_flex, cervical_bend, cervical_rot], axis=1),
                np.stack([thoracic_flex, thoracic_bend, thoracic_rot], axis=1),
                np.stack([lumbar_flex, lumbar_bend, lumbar_rot], axis=1),
                np.stack([sacral_flex, sacral_bend, sacral_rot], axis=1)
            ], axis=1)  # [60, 4, 3]
            
            # Soft-Rigid coupling state (from project book: "毫秒级刚柔切换")
            # Logic: Rigid (1) when high flexion (load bearing), Soft (0) when moving freely
            rigidity = np.zeros((60, 4))
            for seg in range(4):
                for i in range(60):
                    # Rigid state when |angle| > 15° (load bearing) or near extremes
                    if abs(angles[i, seg, 0]) > 15:  # Flexion threshold
                        rigidity[i, seg] = 1.0  # Rigid lock
                    elif abs(angles[i, seg, 1]) > 8:  # Lateral bend threshold
                        rigidity[i, seg] = 0.8  # Partial rigid
                    else:
                        rigidity[i, seg] = 0.2  # Soft/flexible (with noise)
            
            # Generate IMU data from angles (simplified physics)
            # IMU: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            imu = np.zeros((60, 4, 6))
            for seg in range(4):
                # Gyro from angle derivatives
                gyro = np.gradient(angles[:, seg, :], t, axis=0) * np.pi/180  # rad/s
                # Acc from angle projection + gravity (simplified)
                acc = np.zeros((60, 3))
                acc[:, 0] = np.sin(np.radians(angles[:, seg, 0])) * 1.5  # x from flexion
                acc[:, 1] = np.sin(np.radians(angles[:, seg, 1])) * 1.0  # y from bending
                acc[:, 2] = 9.8 * np.cos(np.radians(angles[:, seg, 0]))  # z gravity component
                
                imu[:, seg, :3] = acc + np.random.normal(0, 0.1, (60, 3))
                imu[:, seg, 3:] = gyro + np.random.normal(0, 0.05, (60, 3))
            
            # Extract windows
            input_imu = imu[:self.seq_len]          # [40, 4, 6]
            target_angles = angles[self.seq_len:self.seq_len+self.pred_len]  # [20, 4, 3]
            target_rigidity = rigidity[self.seq_len:self.seq_len+self.pred_len] # [20, 4]
            
            self.samples.append({
                'input': torch.FloatTensor(input_imu),
                'posture': torch.FloatTensor(target_angles),
                'rigidity': torch.FloatTensor(target_rigidity)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    ds = BiomimeticSpineDataset(num_samples=5)
    sample = ds[0]
    print(f"Input IMU: {sample['input'].shape}")       # [40, 4, 6]
    print(f"Target Posture: {sample['posture'].shape}") # [20, 4, 3]
    print(f"Target Rigidity: {sample['rigidity'].shape}") # [20, 4]
