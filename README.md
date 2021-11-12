#  골연령 예측 / [Bone Age Prediction]
### 결과 / [Result]
![image](https://user-images.githubusercontent.com/60537388/141226133-dbff15df-f3db-46f6-9bce-506b361aafb8.png)
## 설명 / [Description]
  소아기의 왼쪽 손목과 손가락 관절이 보이는 x-ray사진을 YoLov5s와 Tjnet 딥러닝을 활용하여 골연령을 예측합니다. 예측된 골연령과 질병관리청에서 제공된 '소아 청소년 성장도표'를 활용하여 18세 기준 예상 신장을 도출합니다. 프로그램 사용을 용이하게 하기 위해 PyQT5를 활용하여 GUI를 구성하였습니다.

  Predict bone age with left hand x-ray images by using deep learning models, YoLov5s and Tjnet. And also predict height when patients get 18 years old with predicted bone age and 'Growth Standards data' by KDCA(Korea Disease Control and Prevention Agency). For using program easily, we made GUI using PyQT5

## 프로젝트 절차 / [Procesure]
### 이미지 전처리 / [Image Preprocessing]
  기존 엑스레이 이미지를 Opencv 모듈을 활용하여 마스크 생성, 배경삭제, 손목을 기준으로 회전, 이진화분류, 뼈 강조 단계를 거쳐 전처리를 완료하였습니다.
  
  Making Mask, removing backgroud, rotatimg image by wrist, classfying by binary, emphasizing bone by using Opencv module step by step.
  Preprocessing x-ray images 

### 라벨링 처리 / [YoLov5 Annotation]
  전처리가 완료된 데이터를 LabelImg 프로그램으로 TW3기법 기반으로 객체를 지정한 후 YoLov5s에 학습시켰고, mAP 측정결과 평균 0.97로 좋은 성능을 보였습니다.
  
  Annotating labels based on Tw3 to completed preprocessed image using LabelImg and fitting to YoLov5s, and reduce loss aobut mAP 0.46

  
  ![image](https://user-images.githubusercontent.com/60537388/141226564-d67390fd-f4a6-4712-8a2f-8b41232f4f7d.png)
  
  <욜로 결과 - 샘플데이터 / [Result YoLov5 - Sample data]>
  (predicting boundary annotations using YoLov5s )

### 딥러닝 모델 학습 / [TJnet]
  도출된 손목과 손가락관절 데이터를 TJnet 모델에 대입하여 학습시킵니다. 특히 Tjnet의 가장 큰 특징인 Gender를 추가하여 성별의 차이 또한 학습하게 했습니다. 전체 데이터가 성별과 연령을 기준으로 크기가 다르기 때문에 학습, 검증 모델은 성별,연령을 기준으로 계층적무작위샘플링 0.7 : 0.3으로 만들었습니다.

  Fitting Crops of wrist and finger joints data into TJnet. Especially inserting gender which Tjnet model's biggest feature for fitting difference of gender. Train and validation dataset were made by using stratified shuffle split (train : 0.7 / validation : 0.3) based on gender and age because total dataset size was irregular.
  
  학습된 TJnet모델 평가 결과, val_loss 는 0.0148 이고, val_month_loss 즉, 전문의 판단과 예측 모델의 판단 결과의 개월수 차이는 약 4.7개월 차이를 보입니다. 

  Result of TJnet models, val_loss is 0.0148 and val_month_loss is 4.7. About model shows 4.7 months difference between average of specialists' diagnosis results.
 
  실제 전문의와의 차이를 시각적으로 보기 위해 블랜드-알트만 그래프를 그렸고, 그래프 해석결과 전문의의 판단과 크게 차이가 없음을 확인했습니다.

  For verifying the difference between speicalists. we draw Blend-Altman graph with TJnet model result and average of specialists' diagnosis results. And the result shows little difference.
  
  ![image](https://user-images.githubusercontent.com/60537388/141226376-4eb96065-7566-49fd-b619-bd0a1b0a6d63.png))

  <두 전문의의 평균값과 모델링 예측값을 이용한 블랜드-알트만 그래프>
  [Blend-Altman graph average of specialists' diagnosis results & TJnet model result]

### 신장 예측 / [LMS Height Prediction]
  이후 예측된 골연령을 활용하여 질병관리청에서 제공된 '소아 청소년 성장도표'를 활용하여 18세 기준 예상 신장을 도출하였습니다. 예측 신장 계산 공식은 수정된 L,M,S 공식을 활용하였고 각각의 값은 L : Box-cox-power, M : Median , S : Coefficient of Variation 입니다. L, M, S의 값은 성장도표에 포함되어 있으며, 해당 연령의 신장과 그에 따른 분위수가 같이 포함되어 있습니다. L, M, S값으로 평균정규분포 (Z)를 구하고, 18세 기준 L, M, S 값과 평균정규분포(Z)로 검사자의 18세 예측 신장값을 도출했습니다. 도출된 예측 신장값을 검사자가 편하게 볼 수 있도록 질병관리청에서 제공한 '소아 청소년 성장그래프'와 동일하게 그래프를 그리고, 그 위에 현재 나이와 신장의 분위수와 예측 신장값을 표시하였습니다.

  Then we predict height of patients when get 18 years old with predicted bone age and 'Growth Standards data' by KDCA(Korea Disease Control and Prevention Agency). Formula of predicting height is from Proc Nlin & Gauss Newton. Each values , L, M, S are box-cox power, Median and Coefficient of variation. These values are contain in provided data also Heights, ages and percentiles. We found Z score with the formula for predicting 18 years height. Then we draw graph same as 'Growth standards graph' by KDCA to point not only patient's current age, height and percentile in black point, but also patient's predicted height in red point.
  
 ![image](https://user-images.githubusercontent.com/60537388/141226282-c36ec03f-d7dc-4030-bf6f-81bd25b541f3.png)
 
   <도출 그래프 / [Growth standards graph]>
 
 
  예측된 골연령과 예측 신장 정보를 쉽게 전문의나 환자가 이용할 수 있게 끔 PyQT5를 이용하여 GUI를 구성하였습니다. 전처리된 파일과 그래프는 각 폴더에 저장이 되도록 하여 정보 유출을 줄였고, 엑셀로 보고서가 저장, 인쇄되게끔 하여 사용자의 편리성을 강조하였습니다. 사용법은 Readme에 적혀있습니다.
  
  To use this program easily, we made GUI programe using PyQT5. Preprocessed file and graph are saved each folders to reduce data loss, and the reports of dignosis are convert to excel formats to save and print for users convenience. Description is in Readme file.
  
 ![image](https://user-images.githubusercontent.com/60537388/141225867-bafa87a5-c773-4942-a95c-269ee07f85ed.png)


# Teammates
  - Kwanghun Kim. https://www.github.com/kikiru328
  - Yubeen Song. https://github.com/yibnn
  - Hyejeong Choi. https://github.com/601chl
  - Seunghey Chae. https://github.com/SeunghyeChae
  - Jonghyeon Lee. https://github.com/Jjongu
  
  
  

