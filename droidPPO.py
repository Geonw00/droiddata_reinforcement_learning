from DroidEnv import DroidEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':
    # 구축한 환경에서 stable_baselines3의 MultiInputPolicy 정책 사용하여 PPO 모델 학습

    # DroidEnv 환경
    # 이용하는 state가 이미지일 경우, DummyVecEnv을 통해 VecTransposeImage를 해야함
    # 사용하지 않을 시, callback 호출 시 evalCallback에서는 유효하지 않은 env로 인식되어 확인이 어려운 이슈 존재
    env = DummyVecEnv([lambda: DroidEnv()])

    # Early Stopping 설정 => 평균 보상
    earlyStopCallback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    # 모델 평가를 위한 EvalCallback 설정
    evalCallback = EvalCallback(env, 
                                # 가장 좋은 모델 저장 경로
                                best_model_save_path='./logs/',
                                log_path='./logs/', 
                                # eval_freq -> value step 주기로 평가  
                                eval_freq=1000,
                                # deterministic=True => 가장 높은 확률 행동을 선택 (평가용)
                                # deterministic=False => 확률에 따른 행동 (학습용)
                                deterministic=True, 
                                render=False)
                                # callback_on_new_best=earlyStopCallback)

    # MlpPolicy => 벡터값
    # CnnPolicy => 이미지값
    # MultiInputPolicy => 벡터, 이미지값
    # PPO 모델 생성
    model = PPO("MultiInputPolicy", env, verbose=0)

    # 모델 학습
    # reward 사용
    model.learn(total_timesteps=50000, callback=evalCallback)
    # model.learn(total_timesteps=1)

    # 학습 완료 후 모델 평가
    rewardMean, rewardStd = evaluate_policy(model, env, n_eval_episodes=10)

    # frames = []

    # observation = env.reset()

    # for _ in range(100):
    #     action, _states = model.predict(observation)
    #     observation, reward, done, info = env.step(action)

    #     # frame = env.render(mode='rgb_array')
    #     # frames.append(frame)

    #     if done:
    #         observation = env.reset()

    env.close()

    # 만든 환경에 맞는 renderer 필요
    # imageio.mimsave("example.mp4", frames, fps=60)