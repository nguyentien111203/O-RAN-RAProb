import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from typing import Dict, Tuple

class ORANPPOAgent:
    def __init__(self, 
                 input_shapes: Dict[str, Tuple[int]],
                 action_dim: int,
                 max_RUs: int = 3,
                 max_users: int = 5,
                 max_RBs: int = 10):
        
        # Tham số hệ thống
        self.max_RUs = max_RUs
        self.max_users = max_users
        self.max_RBs = max_RBs
        
        # Xây dựng policy network
        self.policy = self._build_transformer_policy(input_shapes, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        
        # Tham số PPO
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def _build_transformer_policy(self, 
                                input_shapes: Dict[str, Tuple[int]], 
                                action_dim: int) -> tf.keras.Model:
        """Xây dựng Transformer-based policy network xử lý dữ liệu có sẵn"""
        # Input layers
        inputs = {
            'K': layers.Input(shape=input_shapes['K'], dtype=tf.int32, name='users'),
            'I': layers.Input(shape=input_shapes['I'], dtype=tf.int32, name='RUs'),
            'B': layers.Input(shape=input_shapes['B'], dtype=tf.int32, name='RBs'),
            'H': layers.Input(shape=input_shapes['H'], dtype=tf.float32, name='channel_gains'),
            'Pmax': layers.Input(shape=input_shapes['Pmax'], dtype=tf.float32, name='max_power'),
            'RminK': layers.Input(shape=input_shapes['RminK'], dtype=tf.float32, name='min_rates')
        }
        
        # 1. Embedding các features
        user_emb = layers.Embedding(self.max_users, 64)(inputs['K'])  # [batch, num_users, 64]
        ru_emb = layers.Embedding(self.max_RUs, 64)(inputs['I'])      # [batch, num_RUs, 64]
        
        # 2. Xử lý channel gains với attention
        rb_mask = tf.not_equal(inputs['B'], -1)  # Mask cho RB invalid
        h_norm = tf.where(rb_mask[..., tf.newaxis], inputs['H'], 0.0)  # Apply mask
        
        # Transformer encoder cho channel state
        h_flat = layers.Reshape((-1, input_shapes['H'][-1]))(h_norm)  # [batch, num_RUs*max_RBs, num_users]
        h_encoded = layers.MultiHeadAttention(num_heads=4, key_dim=64)(h_flat, h_flat)
        h_encoded = layers.GlobalAvgPool1D()(h_encoded)  # [batch, num_users]
        
        # 3. Kết hợp features
        user_features = layers.Concatenate()([
            user_emb,
            h_encoded[..., tf.newaxis],
            tf.repeat(inputs['RminK'][..., tf.newaxis], 64, axis=-1)
        ])
        
        ru_features = layers.Concatenate()([
            ru_emb,
            tf.repeat(inputs['Pmax'][..., tf.newaxis], 64, axis=-1)
        ])
        
        # 4. Attention giữa users và RUs
        query = layers.Dense(128)(user_features)
        key = layers.Dense(128)(ru_features)
        value = layers.Dense(128)(ru_features)
        
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(query, key, value)
        x = layers.Concatenate()([user_features, attention])
        x = layers.LayerNormalization()(x)
        
        # 5. Policy và Value heads
        policy_mean = layers.Dense(action_dim, activation='sigmoid')(x)
        policy_logstd = tf.Variable(tf.zeros(action_dim), trainable=True)
        value = layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=[policy_mean, policy_logstd, value])

    def get_action(self, 
                  state: Dict[str, np.ndarray],
                  return_logprob: bool = False) -> Tuple[np.ndarray, dict]:
        """Lấy action từ policy network"""
        # Convert state sang tensor
        state_tensors = {
            k: tf.convert_to_tensor(v[np.newaxis], dtype=tf.int32 if k in ['K', 'I', 'B'] else tf.float32)
            for k, v in state.items()
        }
        
        # Dự đoán action
        mean, logstd, _ = self.policy(state_tensors)
        dist = tfp.distributions.Normal(mean, tf.exp(logstd))
        action = dist.sample()
        
        if return_logprob:
            logprob = dist.log_prob(action)
            return action.numpy()[0], {'logprob': logprob.numpy()[0], 'mean': mean.numpy()[0]}
        return action.numpy()[0]

    def train_step(self, 
                  states: Dict[str, np.ndarray],
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  old_logprobs: np.ndarray,
                  dones: np.ndarray) -> dict:
        """Thực hiện một bước training PPO"""
        # Convert sang tensor
        state_tensors = {
            k: tf.convert_to_tensor(v, dtype=tf.int32 if k in ['K', 'I', 'B'] else tf.float32)
            for k, v in states.items()
        }
        
        with tf.GradientTape() as tape:
            # Tính toán các giá trị mới
            mean, logstd, values = self.policy(state_tensors)
            dist = tfp.distributions.Normal(mean, tf.exp(logstd))
            
            # Tính toán loss
            new_logprobs = dist.log_prob(actions)
            entropy = tf.reduce_mean(dist.entropy())
            
            # Tỉ lệ probability
            ratio = tf.exp(new_logprobs - old_logprobs)
            
            # Advantage estimation
            advantages = self._compute_advantages(rewards, values.numpy(), dones)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Clipped surrogate objective
            min_advantage = tf.where(
                advantages > 0,
                (1 + self.clip_ratio) * advantages,
                (1 - self.clip_ratio) * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))
            
            # Value loss
            value_loss = tf.reduce_mean((rewards - values)**2)
            
            # Tổng loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Gradient update
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        return {
            'total_loss': loss.numpy(),
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy': entropy.numpy()
        }

    def _compute_advantages(self,
                           rewards: np.ndarray,
                           values: np.ndarray,
                           dones: np.ndarray) -> np.ndarray:
        """Tính toán Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = values[t+1]
                next_non_terminal = 1 - dones[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        return advantages

# Ví dụ sử dụng với dữ liệu có sẵn
if __name__ == "__main__":
    # Giả lập dữ liệu đầu vào (thay bằng dữ liệu thực của bạn)
    sample_state = {
        'K': np.array([0, 1, 2]),  # Danh sách users
        'I': np.array([0, 1, 2]),  # Danh sách RUs
        'B': np.array([[0, 1, 2, -1], [0, 1, -1, -1], [0, 1, 2, 3]]),  # RB allocation (pad -1)
        'H': np.random.uniform(0.1, 1.0, size=(3, 4, 3)),  # Channel gains (RU, RB, User)
        'Pmax': np.array([300.0, 300.0, 300.0]),  # Max power per RU
        'RminK': np.array([2.0, 10.0, 6.0])  # Min rate per user (Mbps)
    }
    
    # Khởi tạo agent
    input_shapes = {k: v.shape for k, v in sample_state.items()}
    agent = ORANPPOAgent(input_shapes, action_dim=6)  # action_dim = num_users + num_RUs
    
    # Lấy action mẫu
    action, action_info = agent.get_action(sample_state, return_logprob=True)
    print("Generated action:", action)
    
    # Giả lập training (thay bằng dữ liệu thực)
    batch_states = {k: np.repeat(v[np.newaxis], 5, axis=0) for k, v in sample_state.items()}
    batch_actions = np.random.uniform(0, 1, size=(5, 6))
    batch_rewards = np.random.normal(1.0, 0.5, size=5)
    batch_logprobs = np.random.normal(0, 1, size=5)
    batch_dones = np.array([0, 0, 0, 1, 0])
    
    # Training step
    metrics = agent.train_step(batch_states, batch_actions, batch_rewards, batch_logprobs, batch_dones)
    print("Training metrics:", metrics)