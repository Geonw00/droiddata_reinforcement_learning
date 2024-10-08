import os
import h5py
import numpy as np

# Droid데이터 셋 환경
class DroidDataProcessor():
    def __init__(self):
        print('init DroidDataProcessor')

    # 파일 경로 추출
    def findPath(self):
        labN = ['PennPAL', 'IRIS', 'GuptaLab', 'AUTOLab', 'ILIAD', 'RPL', 
                'RAD', 'CLVR', 'REAL', 'WEIRD', 'IPRL', 'TRI', 'RAIL']

        existH5List = []
        for lab in labN:
            # success, failure 폴더 존재
            labDataPath = f'파일 경로{lab}'
            try:
                for ffn in os.listdir(labDataPath):
                    for sfn in os.listdir(f'{labDataPath}/{ffn}'):
                        # AUTOLab 폴더 failure에 svo_names.json 존재 예외 처리
                        if sfn.endswith('.json'):
                            continue

                        for tfn in os.listdir(f'{labDataPath}/{ffn}/{sfn}'):
                            for inFile in os.listdir(f'{labDataPath}/{ffn}/{sfn}/{tfn}'):
                                # im128.h5 파일일 경우
                                # image에 대한 벡터값은 im128.h5에만 존재
                                if inFile == 'trajectory_im128.h5':
                                    if '파일 경로' != f'{labDataPath}/{ffn}/{sfn}/{tfn}':
                                        existH5List.append(f'{labDataPath}/{ffn}/{sfn}/{tfn}')
                                
            except Exception as e:
                print(f'findPath Error : {e}')

        # 병합 리스트 반환
        return existH5List[:100]

    # 에피소드 h5 추출
    def loadH5(self, path):
        try:
            with h5py.File(f'{path}/trajectory_im128.h5', 'r') as h5:
                readH5 = self.mergeH5(h5, '', 0, {})
        except Exception as e:
            # print(f'loadH5 Error : {e}')
            pass
        return readH5

    # h5 파일 Dict형태로 변환하는 재귀 함수
    def mergeH5(self, inFile, path, depth, h5Dict):
        # Input/Output error의 원인이 메모리문제라고 생각하여 필요하지 않은 벡터값 제거
        passList = ['hand_camera_right_image', 'varied_camera_1_right_image', 'varied_camera_2_right_image']
        h5DictResult = h5Dict
        try:
            if type(inFile) == h5py._hl.group.Group or type(inFile) == h5py._hl.files.File:
                inKeyList = inFile.keys()
                for k in inKeyList:
                    if depth == 1 and path.split('/')[1] not in h5DictResult:
                        h5DictResult[path.split('/')[1]] = {}
                    elif depth == 2:
                        path1, path2 = path.split('/')[1:]
                        if path2 not in h5DictResult[path1]:
                            h5DictResult[path1][path2] = {}
                    elif depth == 3:
                        path1, path2, path3 = path.split('/')[1:]
                        if path3 not in h5DictResult[path1][path2]:
                            h5DictResult[path1][path2][path3] = {}

                    # 재귀
                    h5DictResult = self.mergeH5(inFile[k], path + '/' + k, depth+1, h5DictResult)

            # 하위 필드 병합
            else:
                if depth == 2:
                    path1, path2 = path.split('/')[1:]
                    try:
                        h5DictResult[path1][path2] = inFile[()].tolist()
                    except:
                        h5DictResult[path1][path2] = inFile.tolist()

                elif depth == 3:
                    path1, path2, path3 = path.split('/')[1:]
                    try:
                        h5DictResult[path1][path2][path3] = inFile[()].tolist()
                    except:
                        h5DictResult[path1][path2][path3] = inFile.tolist()
                
                elif depth == 4:
                    path1, path2, path3, path4 = path.split('/')[1:]
                    if path4 in passList:
                        pass
                    else:
                        try:
                            h5DictResult[path1][path2][path3][path4] = inFile[()].tolist()
                        except Exception as err:
                            h5DictResult[path1][path2][path3][path4] = inFile.tolist()
        except Exception as e:
            # print(f'mergeH5 Error : {e}')
            pass

        return h5DictResult

    # target => inH5['action']['target_cartesian_position']
    # action => inH5['action']['abs_pos'] + inH5['action']['gripper_position']
    # image => inH5['observation']['camera']['image']['varied_camera_1_left_image']

    # state 및 image 데이터 전처리
    def preprocessingData(self, h5File):
        resImages = []
        resStates = []
        for image, target in zip(h5File['observation']['camera']['image']['varied_camera_1_left_image'], h5File['action']['target_cartesian_position']):
            resImages.append(np.array(image))
            resStates.append(target[:3])
        
        return resImages, resStates