from DroidDataProcessor import DroidDataProcessor as dp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# 환경 구축 (gym 기반)

# Droid데이터 셋 환경
class DroidEnv(gym.Env):
    def __init__(self):
        super(DroidEnv, self).__init__()
        
        # im128.h5 파일 경로 리스트 추출 
        self.oriH5FilePathList = dp.findPath()
        self.h5FilePathList = self.oriH5FilePathList.copy()
        self.images, self.states = dp.preprocessingData(dp.loadH5('파일 경로'))
        self.currentStep = 0
        self.done = False
        self.maxSteps = len(self.images) 
        self.count = 1

        # observation_space (Image만 사용할 경우)
        # self.observation_space = spaces.Box(low=0, high=255, shape=self.images.shape, dtype=np.uint8)

        # observation_space (Image와 state 사용할 경우)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=self.images[0].shape, dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        # action_space (x, y, z, w)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # 데이터 셋 재순환
    def resetDataset(self):
        self.h5FilePathList = self.oriH5FilePathList.copy()
        self.images, self.states = dp.preprocessingData(dp.loadH5('파일 경로'))
        self.maxSteps = len(self.images) 
        self.count = 1
        self.currentStep = 0

    # 초기화
    def reset(self, seed=805):
        try:
            # 한 에피소드가 끝나면 다음 에피소드 진행
            if len(self.h5FilePathList) > 0:
                inH5 = dp.loadH5(self.h5FilePathList.pop())
                self.images, self.states = dp.preprocessingData(inH5)
                # print('here')
                self.maxSteps = len(self.images) 
                self.count += 1
                self.currentStep = 0
                if self.maxSteps == 0:
                    self.reset()

                #     self.resetDataset()
            # print(len(self.images))
        except Exception as e:
            self.reset()

        self.currentStep = 0

        return self.getObservation(), {}

    def step(self, action):
        # if self.maxSteps == 0:
        #     self.done = True
        #     return observation, reward, self.done, {}, {} 
        
        self.currentStep += 1
        # 하나의 에피소드 데이터 기준 
        if self.currentStep >= self.maxSteps:
            self.done = True

        # reward
        state = self.states[self.currentStep - 1]
        reward = self.makeReward(action, state)
        
        # episode가 끝나면 초기화
        if self.done:
            observation = self.reset()
        else:
            observation = self.getObservation()

        return observation, reward, self.done, {}, {}
    
    # Droid 데이터 셋의 경우, reward가 없기 때문에 만들어 줘야 함
    # xarm의 reward 및 robomic의 reward 참고
    # reward
    def makeReward(self, action, target):
        # 그리퍼의 x, y, z
        gripperPosition = action[:3]

        # 타겟 위치 x, y, z
        targetPosition = target  

        # 그리퍼와 타겟 간의 거리
        xDis = gripperPosition[0] - targetPosition[0]
        yDis = gripperPosition[1] - targetPosition[1]
        zDis = gripperPosition[2] - targetPosition[2]

        # 그리퍼와 타겟의 거리가 가까울 수록 보상 증가
        avgDis = (xDis + yDis + zDis) / 3
        reward = -avgDis

        # xarm z_threshold => 0.15
        # FetchPickAndPlaceDense => 0.05
        # 물체를 잡았을 경우
        if xDis < 0.05 and yDis < 0.05 and zDis < 0.15:
            reward += 4
        
        return reward

    def getObservation(self):
        # image = np.transpose(self.images[self.currentStep], (2, 0, 1))
        return {
            "image": np.array(self.images[self.currentStep-1]),
            "state": self.states[self.currentStep-1]
        }
    
    # 렌더러 구현 필요
    def render(self, mode):
        try:
            ar = MujocoRenderer.render(self, mode, camera_name="camera2")
        except Exception as e:
            print(e)
        return ar