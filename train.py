import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------- 1. Student Model 정의 -----------------
class StudentPolicy(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super().__init__()
        # Teacher 대비 훨씬 작은 CNN 백본 사용
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ... 레이어 개수와 채널 수를 줄임
        )
        # 마지막은 로봇 동작(Action)을 출력하는 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=..., out_features=128), # features_in 크기는 CNN 출력 크기에 맞춤
            nn.ReLU(),
            nn.Linear(128, action_dim) # action_dim: 예: (x, y, z, roll, pitch, yaw) 6차원
        )

    def forward(self, x):
        features = self.cnn_backbone(x)
        flattened = torch.flatten(features, 1)
        action = self.mlp(flattened)
        return action

# ----------------- 2. 모방 학습 데이터셋 -----------------
class ImitationDataset(Dataset):
    def __init__(self, demonstrations):
        # demonstrations: (observation, teacher_action) 쌍의 리스트
        self.observations = demonstrations['obs']
        self.teacher_actions = demonstrations['action']

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.teacher_actions[idx]


# ----------------- 3. 학습 설정 -----------------
# 1. 모델 인스턴스화
# teacher_policy = TeacherPolicy(...) # (학습이 완료된 Teacher 모델)
student_policy = StudentPolicy(observation_shape=(3, 224, 224), action_dim=6)

# 2. 데이터 로더
# dataset = ImitationDataset(loaded_teacher_demonstrations)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. 최적화 함수 및 손실 함수
optimizer = torch.optim.Adam(student_policy.parameters(), lr=1e-4)

# 모방 학습 손실: Student의 예측 행동이 Teacher의 행동과 얼마나 가까운가? (MSE 사용)
criterion = nn.MSELoss() 

# ----------------- 4. 학습 루프 -----------------
def train_student(student_policy, dataloader, criterion, optimizer, num_epochs=10):
    student_policy.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (obs, teacher_action) in enumerate(dataloader):
            
            # obs: 이미지, teacher_action: Teacher가 내린 정답 행동
            
            # 1. 순전파
            student_action_pred = student_policy(obs)

            # 2. 손실 계산: Student 행동과 Teacher 행동 간의 오차
            loss = criterion(student_action_pred, teacher_action)
            
            # 3. 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# train_student(student_policy, dataloader, criterion, optimizer)
