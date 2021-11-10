#  골연령 예측 / [Bone Age Prediction]
### 결과 / [Result]
![image](https://user-images.githubusercontent.com/60537388/141145083-3da66042-19f0-482e-a720-18a6b910ceda.png)
## 설명 / [Description]
소아기의 왼쪽 손목과 손가락 관절이 보이는 x-ray사진을 YoLov5s와 Tjnet 딥러닝을 활용하여 골연령을 예측합니다. 예측된 골연령과 질병관리청에서 제공된 '소아 청소년 성장도표'를 활용하여 18세 기준 예상 신장을 도출합니다. 프로그램 사용을 용이하게 하기 위해 PyQT5를 활용하여 GUI를 구성하였습니다.

Predict bone age with left hand x-ray images by using deep learning models, YoLov5s and Tjnet. And also predict height when patients get 18 years old with predicted bone age and 'Growth Standards data' by KDCA(Korea Disease Control and Prevention Agency). For using program easily, we made GUI using PyQT5

## 프로젝트 절차 / [Procesure]
- 이미지 전처리 / Image Preprocessing
  기존 엑스레이 이미지를 Opencv 모듈을 활용하여 마스크 생성, 배경삭제, 손목을 기준으로 회전, 이진화분류, 뼈 강조 단계를 거쳐 전처리를 완료하였습니다.
  
  Making Mask, removing backgroud, rotatimg image by wrist, classfying by binary, emphasizing bone by using Opencv module step by step.
  Preprocessing x-ray images 

- 전처리가 완료된 데이터를 LabelImg 프로그램으로 TW3기법 기반으로 객체를 지정한 후 YoLov5s에 학습시켰고, mAP가 0.46 정도 손실을 줄였습니다
  
  Annotating labels based on Tw3 to completed preprocessed image using LabelImg and fitting to YoLov5s, and reduce loss aobut mAP 0.46

  욜로 결과 / [YoLov5]
  ![image](https://user-images.githubusercontent.com/60537388/141142438-bab0f93a-472e-4ae8-849b-1e95b8cb0838.png)
  
  (predicting boundary annotations using YoLov5s )

- 도출된 손목과 손가락관절 데이터를 TJnet 모델에 대입하여 학습시킵니다. 특히 Tjnet의 가장 큰 특징인 Gender를 추가하여 성별의 차이 또한 학습하게 했습니다.

  Fitting Crops of wrist and finger joints data into TJnet. Especially inserting gender which Tjnet model's biggest feature for fitting difference of gender.
  
- 
  
 
최혜정 송유빈 채승혜 이종현 김광훈

