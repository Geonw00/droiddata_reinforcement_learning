# droiddata_reinforcement_learning

- Droid 데이터 셋 활용
- 강화학습 공부

- 로봇 학습용 데이터
  - 실제 세계에서 로봇을 통한 물리적인 데이터
  - 문제 : 로봇 구매 등 물리적인 환경 구성에 막대한 자금 소요로 규모의 경제, 높은 진입장벽 등 문제 발생

- 가상 환경을 통한 데이터 확보 문제 해결
  - LeRobot
    - 허깅페이스에서 공개한 오픈 소스 로봇 개발 도구
    - 가상 환경을 통해 하드웨어(로봇) 없이도 AI모델을 시뮬레이션 및 테스트 가능
   
  - 대표적인 예시
    - 구글 웨이모(Waymo)
    - 자율주행 학습 데이터 확보를 위해 카 크래프트 시뮬레이터를 활용하여 강화학습을 통한 데이터 증강
   
- 가상 환경 생성
  - 기존의 가상 환경 비교
    - Gym-Xarm
      - 각 State 요소의 정의를 정확하게 알 수 없음
    - Gym-Aloha
      - Aloha 가상 환경은 팔이 두 개인 로봇을 기준으로 만든 가상 환경
      - Droid의 경우, Single Arm 형태
    - FetchPickAndPlace-v2
      - State 요소에 대한 정의 및 Single Arm을 다루는 가상 환경
      - 이미지(RGB) 값 활용 x

  - 기존의 가상 환경들을 기반으로 이미지 값을 활용하는 가상 환경 생성
    - OpenAI의 Gym 라이브러리 기반
  - A2C, PPO 알고리즘 중 A2C의 단점을 개선한 PPO 사용
  - 벡터, 이미지 값을 모두 활용하기 위해 Stable-baselines3 기반 MultiInputPolicy 사용
